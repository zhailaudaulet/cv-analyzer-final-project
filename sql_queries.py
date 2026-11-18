"""
Analytical SQL queries for the Resume Analyzer database (requirement 10).

Each query is a dict with a short title, what it answers, the SQL techniques
it demonstrates, and the SQL text. Run them all with:

    python sql_queries.py
or load a single result into pandas:

    from sql_queries import QUERIES, run_named
    df = run_named("top_resume_skills")
"""
from database import run_query, DEFAULT_DB

QUERIES = {
    # 1 ─────────────────────────────────────────────────────────────────────
    "top_resume_skills": {
        "title": "Most common skills in resumes",
        "question": "Which skills do candidates list most often?",
        "techniques": ["JOIN", "GROUP BY", "ORDER BY/LIMIT"],
        "sql": """
            SELECT s.name AS skill,
                   COUNT(*) AS resumes,
                   ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM resumes), 1) AS pct_of_resumes
            FROM resume_skills rs
            JOIN skills s ON s.skill_id = rs.skill_id
            GROUP BY s.name
            ORDER BY resumes DESC
            LIMIT 20;
        """,
    },
    # 2 ─────────────────────────────────────────────────────────────────────
    "skill_supply_vs_demand": {
        "title": "Skill gap: demand from jobs vs supply in resumes",
        "question": "Which skills are required by several jobs (dataset postings + the app's 5 vacancies) "
                    "but rare among candidates?",
        "techniques": ["CTE", "LEFT JOIN", "GROUP BY/HAVING", "COALESCE"],
        "sql": """
            WITH demand AS (
                SELECT skill_id, COUNT(*) AS jobs_requiring
                FROM job_skills
                GROUP BY skill_id
                HAVING COUNT(*) >= 2
            ),
            supply AS (
                SELECT skill_id, COUNT(*) AS resumes_with_skill
                FROM resume_skills
                GROUP BY skill_id
            )
            SELECT s.name AS skill,
                   d.jobs_requiring,
                   COALESCE(sp.resumes_with_skill, 0) AS resumes_with_skill,
                   ROUND(100.0 * COALESCE(sp.resumes_with_skill, 0)
                         / (SELECT COUNT(*) FROM resumes), 1) AS pct_resumes
            FROM demand d
            JOIN skills s       ON s.skill_id = d.skill_id
            LEFT JOIN supply sp ON sp.skill_id = d.skill_id
            ORDER BY d.jobs_requiring DESC, resumes_with_skill ASC;
        """,
    },
    # 3 ─────────────────────────────────────────────────────────────────────
    "job_score_ranking": {
        "title": "Average match score per job, ranked",
        "question": "Which job postings are easiest / hardest to match in the candidate pool?",
        "techniques": ["JOIN", "GROUP BY/HAVING", "window RANK()"],
        "sql": """
            SELECT j.title,
                   COUNT(*)                         AS candidates,
                   ROUND(AVG(m.matched_score), 3)   AS avg_score,
                   ROUND(MAX(m.matched_score), 3)   AS best_score,
                   SUM(m.matched_score >= 0.7)      AS strong_matches,
                   RANK() OVER (ORDER BY AVG(m.matched_score) DESC) AS rank_by_avg
            FROM matches m
            JOIN jobs j ON j.job_id = m.job_id
            GROUP BY j.job_id
            HAVING COUNT(*) >= 300
            ORDER BY rank_by_avg;
        """,
    },
    # 4 ─────────────────────────────────────────────────────────────────────
    "top3_jobs_per_resume": {
        "title": "Top-3 recommended jobs for each candidate",
        "question": "For every resume, which three jobs fit best according to the labels?",
        "techniques": ["CTE", "window ROW_NUMBER() PARTITION BY", "JOIN"],
        "sql": """
            WITH ranked AS (
                SELECT m.resume_id, m.job_id, m.matched_score,
                       ROW_NUMBER() OVER (PARTITION BY m.resume_id
                                          ORDER BY m.matched_score DESC, m.job_id) AS rn
                FROM matches m
            )
            SELECT r.resume_id, r.rn AS position, j.title, ROUND(r.matched_score, 3) AS matched_score
            FROM ranked r
            JOIN jobs j ON j.job_id = r.job_id
            WHERE r.rn <= 3
            ORDER BY r.resume_id, r.rn;
        """,
    },
    # 5 ─────────────────────────────────────────────────────────────────────
    "sql_skill_overlap_baseline": {
        "title": "Skill-overlap score computed in SQL vs the ground-truth label",
        "question": "How well does a pure SQL skill-overlap baseline agree with matched_score?",
        "techniques": ["multiple CTEs", "LEFT JOIN", "GROUP BY", "CASE"],
        "sql": """
            WITH job_size AS (
                SELECT job_id, COUNT(*) AS n_required
                FROM job_skills
                GROUP BY job_id
            ),
            overlap AS (
                SELECT m.resume_id, m.job_id, COUNT(js.skill_id) AS n_matched
                FROM matches m
                LEFT JOIN resume_skills rs ON rs.resume_id = m.resume_id
                LEFT JOIN job_skills js    ON js.job_id = m.job_id AND js.skill_id = rs.skill_id
                GROUP BY m.resume_id, m.job_id
            )
            SELECT o.resume_id, o.job_id, j.title,
                   o.n_matched, js.n_required,
                   ROUND(1.0 * o.n_matched / js.n_required, 3) AS overlap_score,
                   m.matched_score,
                   CASE WHEN m.matched_score >= 0.7 THEN 'strong'
                        WHEN m.matched_score >= 0.5 THEN 'medium'
                        ELSE 'weak' END AS label
            FROM overlap o
            JOIN job_size js ON js.job_id = o.job_id
            JOIN matches m   ON m.resume_id = o.resume_id AND m.job_id = o.job_id
            JOIN jobs j      ON j.job_id = o.job_id;
        """,
    },
    # 6 ─────────────────────────────────────────────────────────────────────
    "above_average_candidates": {
        "title": "Candidates whose average score beats the overall average",
        "question": "Which resumes are strong across many jobs, and how many skills do they list?",
        "techniques": ["subquery in HAVING", "correlated subquery", "GROUP BY"],
        "sql": """
            SELECT m.resume_id,
                   ROUND(AVG(m.matched_score), 3) AS avg_score,
                   (SELECT COUNT(*) FROM resume_skills rs
                     WHERE rs.resume_id = m.resume_id) AS n_skills,
                   (SELECT COUNT(*) FROM experience e
                     WHERE e.resume_id = m.resume_id) AS n_positions
            FROM matches m
            GROUP BY m.resume_id
            HAVING AVG(m.matched_score) > (SELECT AVG(matched_score) FROM matches)
            ORDER BY avg_score DESC
            LIMIT 25;
        """,
    },
    # 7 ─────────────────────────────────────────────────────────────────────
    "score_by_experience_requirement": {
        "title": "Match scores by required experience, with cumulative share",
        "question": "Do jobs that ask for more experience get lower match scores?",
        "techniques": ["CASE bucketing", "GROUP BY", "window SUM() OVER (running total)"],
        "sql": """
            WITH bucketed AS (
                SELECT CASE
                           WHEN j.min_experience_years IS NULL THEN '0. not stated'
                           WHEN j.min_experience_years <= 1   THEN '1. 0-1 years'
                           WHEN j.min_experience_years <= 3   THEN '2. 2-3 years'
                           WHEN j.min_experience_years <= 5   THEN '3. 4-5 years'
                           ELSE '4. 6+ years'
                       END AS experience_bucket,
                       m.matched_score
                FROM matches m
                JOIN jobs j ON j.job_id = m.job_id
            )
            SELECT experience_bucket,
                   COUNT(*) AS pairs,
                   ROUND(AVG(matched_score), 3) AS avg_score,
                   ROUND(100.0 * SUM(COUNT(*)) OVER (ORDER BY experience_bucket)
                         / SUM(COUNT(*)) OVER (), 1) AS cumulative_pct
            FROM bucketed
            GROUP BY experience_bucket
            ORDER BY experience_bucket;
        """,
    },
    # 8 ─────────────────────────────────────────────────────────────────────
    "candidate_percentile_per_job": {
        "title": "Where a candidate stands among all applicants for each job",
        "question": "For each job, who is in the top 10% and how far are they above the job average?",
        "techniques": ["window NTILE()", "PERCENT_RANK()", "AVG() OVER (PARTITION BY)", "subquery"],
        "sql": """
            SELECT title, resume_id, matched_score, decile,
                   ROUND(pct_rank, 3) AS percent_rank,
                   ROUND(matched_score - job_avg, 3) AS above_job_avg
            FROM (
                SELECT j.title, m.resume_id, m.matched_score,
                       NTILE(10)      OVER w AS decile,
                       PERCENT_RANK() OVER w AS pct_rank,
                       AVG(m.matched_score) OVER (PARTITION BY m.job_id) AS job_avg
                FROM matches m
                JOIN jobs j ON j.job_id = m.job_id
                WINDOW w AS (PARTITION BY m.job_id ORDER BY m.matched_score DESC)
            )
            WHERE decile = 1
            ORDER BY title, matched_score DESC;
        """,
    },
    # 9 ─────────────────────────────────────────────────────────────────────
    "skill_cooccurrence": {
        "title": "Skills that appear together in resumes",
        "question": "Which pairs of skills are most often listed together?",
        "techniques": ["self-JOIN", "GROUP BY/HAVING"],
        "sql": """
            SELECT s1.name AS skill_a, s2.name AS skill_b, COUNT(*) AS resumes_with_both
            FROM resume_skills a
            JOIN resume_skills b ON b.resume_id = a.resume_id AND b.skill_id > a.skill_id
            JOIN skills s1 ON s1.skill_id = a.skill_id
            JOIN skills s2 ON s2.skill_id = b.skill_id
            GROUP BY a.skill_id, b.skill_id
            HAVING COUNT(*) >= 20
            ORDER BY resumes_with_both DESC
            LIMIT 20;
        """,
    },
    # 10 ────────────────────────────────────────────────────────────────────
    "app_analysis_history": {
        "title": "Analyses run in the web app (latest first, per vacancy)",
        "question": "What did users analyse in the app, and which skills were most often missing?",
        "techniques": ["JOIN", "LEFT JOIN", "window ROW_NUMBER()", "GROUP BY"],
        "sql": """
            WITH missing AS (
                SELECT a.job_id, s.name AS skill, COUNT(*) AS times_missing,
                       ROW_NUMBER() OVER (PARTITION BY a.job_id ORDER BY COUNT(*) DESC) AS rn
                FROM analysis_skills ask
                JOIN analysis_results a ON a.analysis_id = ask.analysis_id
                JOIN skills s           ON s.skill_id = ask.skill_id
                WHERE ask.status = 'missing'
                GROUP BY a.job_id, s.name
            )
            SELECT j.title, COUNT(a.analysis_id) AS analyses,
                   ROUND(AVG(a.final_score), 3) AS avg_final_score,
                   MAX(a.created_at) AS last_run,
                   (SELECT GROUP_CONCAT(skill, ', ') FROM missing mi
                     WHERE mi.job_id = j.job_id AND mi.rn <= 3) AS most_missing_skills
            FROM jobs j
            LEFT JOIN analysis_results a ON a.job_id = j.job_id
            WHERE j.source = 'app'
            GROUP BY j.job_id
            ORDER BY analyses DESC;
        """,
    },
}


def run_named(name: str, db_path=DEFAULT_DB):
    """Run one query from QUERIES by name and return a pandas DataFrame."""
    if name not in QUERIES:
        raise KeyError(f"Unknown query '{name}'. Available: {', '.join(QUERIES)}")
    return run_query(QUERIES[name]["sql"], db_path)


if __name__ == "__main__":
    for i, (name, q) in enumerate(QUERIES.items(), start=1):
        df = run_named(name)
        print(f"\n[{i}] {q['title']}  ({', '.join(q['techniques'])})")
        print(df.head(8).to_string(index=False))
