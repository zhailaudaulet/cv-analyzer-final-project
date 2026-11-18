"""
SQLite storage for the Resume Analyzer.

Builds a normalized database (see db/schema.sql) from data/resume_data.csv:
resumes, jobs, skills, resume/job skill links, education, experience,
ground-truth match scores, and the results produced by the web app.

Usage:
    python database.py                      # build resume_analyzer.db
    python database.py --csv path/to.csv --db path/to.db
"""
import argparse
import ast
import json
import re
import sqlite3
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "resume_data.csv"
DEFAULT_DB = BASE_DIR / "resume_analyzer.db"
SCHEMA_FILE = BASE_DIR / "db" / "schema.sql"

# Columns that describe the candidate. Every resume is repeated once per job
# posting in the CSV, so these columns together identify a unique resume.

RESUME_COLUMNS = [
    "address", "career_objective", "skills", "educational_institution_name",
    "degree_names", "passing_years", "educational_results", "result_types",
    "major_field_of_studies", "professional_company_names", "company_urls",
    "start_dates", "end_dates", "related_skils_in_job", "positions", "locations",
    "extra_curricular_activity_types", "extra_curricular_organization_names",
    "extra_curricular_organization_links", "role_positions", "languages",
    "proficiency_levels", "certification_providers", "certification_skills",
    "online_links", "issue_dates", "expiry_dates",
]

SKILL_ALIASES = {
    "js": "javascript", "py": "python", "ml": "machine learning",
    "dl": "deep learning", "nlp": "natural language processing",
    "tf": "tensorflow", "k8s": "kubernetes", "oop": "object oriented programming",
    "reactjs": "react", "react.js": "react", "nodejs": "node.js",
    "postgres": "postgresql", "ms excel": "excel", "microsoft excel": "excel",
}
EMPTY_VALUES = {"", "n/a", "na", "none", "null", "nan", "-"}


# ── Parsing helpers ───────────────────────────────────────────────────────────

def load_resume_csv(csv_path=DEFAULT_CSV) -> pd.DataFrame:
    """Read the dataset and fix the stray byte-order mark in one column name."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise SystemExit(
            f"Dataset not found: {csv_path}\n"
            "Download it from https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset "
            "and put resume_data.csv into the data/ folder."
        )
    # The header contains '﻿job_position_name' (a BOM in the middle of the file)
    df.columns = [c.replace("﻿", "").strip() for c in df.columns]
    return df


def parse_list(raw) -> list:
    """Turn a list stored as text ("['Python', 'SQL']", nested lists, plain text) into a flat list."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = [part.strip(" '\"") for part in raw.strip("[]").split(",")]
    else:
        value = [raw]

    flat = []
    stack = list(value) if isinstance(value, (list, tuple)) else [value]
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack[0:0] = list(item)          # flatten nested lists, keep order
        elif item is None:
            flat.append(None)
        else:
            flat.append(str(item).strip())
    return flat


def clean_value(value):
    """Return None for empty placeholders such as 'N/A' or 'None'."""
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in EMPTY_VALUES else value


def normalize_skill(name) -> str | None:
    """Lowercase, trim, collapse spaces and resolve aliases. Returns None for junk."""
    name = clean_value(name)
    if name is None:
        return None
    name = re.sub(r"\s+", " ", name.lower()).strip(" .,;:-•")
    name = SKILL_ALIASES.get(name, name)
    if len(name) < 2 and name not in {"r", "c"}:
        return None
    return name


def split_skill_items(items) -> list[str]:
    """Some list items hold several skills ('excel, word; outlook') — split them."""
    parts = []
    for item in items:
        if isinstance(item, str):
            parts.extend(re.split(r"[,;]", item))
    return parts


def parse_min_years(text) -> int | None:
    """'At least 3 years' -> 3, '5 to 8 years' -> 5, missing -> None."""
    if not isinstance(text, str):
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


# ── Building the tables ───────────────────────────────────────────────────────

def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the flat CSV into normalized DataFrames, one per database table."""
    df = df.copy()

    # Unique resumes / unique jobs
    resume_key = df[RESUME_COLUMNS].fillna("").astype(str).apply("|".join, axis=1)
    df["resume_id"] = pd.factorize(resume_key)[0] + 1
    df["job_title_clean"] = df["job_position_name"].str.strip()
    df["job_id"] = pd.factorize(df["job_title_clean"])[0] + 1

    resume_rows = df.drop_duplicates("resume_id")
    job_rows = df.drop_duplicates("job_id")

    resumes = pd.DataFrame({
        "resume_id": resume_rows["resume_id"],
        "address": resume_rows["address"].map(clean_value),
        "career_objective": resume_rows["career_objective"].map(clean_value),
        "languages": resume_rows["languages"].map(
            lambda raw: ", ".join(v for v in parse_list(raw) if clean_value(v)) or None),
        "skills_raw": resume_rows["skills"],
    })

    jobs = pd.DataFrame({
        "job_id": job_rows["job_id"],
        "job_key": None,
        "source": "dataset",
        "title": job_rows["job_title_clean"],
        "education_requirement": job_rows["educationaL_requirements"].map(clean_value),
        "experience_requirement": job_rows["experiencere_requirement"].map(clean_value),
        "min_experience_years": job_rows["experiencere_requirement"].map(parse_min_years),
        "age_requirement": job_rows["age_requirement"].map(clean_value),
        "responsibilities": job_rows["responsibilities.1"].map(clean_value),
    })

    # Skills of every resume and every dataset job
    resume_skill_pairs = {
        (rid, skill)
        for rid, raw in zip(resume_rows["resume_id"], resume_rows["skills"])
        for skill in map(normalize_skill, split_skill_items(parse_list(raw))) if skill
    }
    job_skill_pairs = {
        (jid, skill)
        for jid, raw in zip(job_rows["job_id"], job_rows["skills_required"])
        if isinstance(raw, str)
        for skill in map(normalize_skill, raw.split("\n")) if skill
    }

    # The 5 vacancies shown in the web app (jobs.py)
    try:
        from jobs import JOBS
    except ImportError:
        JOBS = {}
    next_id = int(jobs["job_id"].max()) + 1
    app_jobs = []
    for offset, job in enumerate(JOBS.values()):
        jid = next_id + offset
        app_jobs.append({
            "job_id": jid, "job_key": job.id, "source": "app", "title": job.title,
            "education_requirement": None, "experience_requirement": job.level,
            "min_experience_years": None, "age_requirement": None,
            "responsibilities": "\n".join(job.responsibilities),
        })
        job_skill_pairs |= {(jid, s) for s in map(normalize_skill, job.skills) if s}
    if app_jobs:
        jobs = pd.concat([jobs, pd.DataFrame(app_jobs)], ignore_index=True)

    all_skills = sorted({s for _, s in resume_skill_pairs} | {s for _, s in job_skill_pairs})
    skills = pd.DataFrame({"skill_id": range(1, len(all_skills) + 1), "name": all_skills})
    skill_ids = dict(zip(skills["name"], skills["skill_id"]))

    resume_skills = pd.DataFrame(
        sorted((rid, skill_ids[s]) for rid, s in resume_skill_pairs),
        columns=["resume_id", "skill_id"])
    job_skills = pd.DataFrame(
        sorted((jid, skill_ids[s]) for jid, s in job_skill_pairs),
        columns=["job_id", "skill_id"])

    # Education and experience: the CSV stores them as parallel lists
    education, experience = [], []
    for _, row in resume_rows.iterrows():
        for inst, degree, major, year, result in zip_longest(
                parse_list(row["educational_institution_name"]), parse_list(row["degree_names"]),
                parse_list(row["major_field_of_studies"]), parse_list(row["passing_years"]),
                parse_list(row["educational_results"])):
            if any(clean_value(v) for v in (inst, degree, major)):
                education.append((row["resume_id"], clean_value(inst), clean_value(degree),
                                  clean_value(major), clean_value(year), clean_value(result)))
        for company, position, start, end in zip_longest(
                parse_list(row["professional_company_names"]), parse_list(row["positions"]),
                parse_list(row["start_dates"]), parse_list(row["end_dates"])):
            if clean_value(company) or clean_value(position):
                experience.append((row["resume_id"], clean_value(company), clean_value(position),
                                   clean_value(start), clean_value(end)))

    education = pd.DataFrame(education, columns=[
        "resume_id", "institution", "degree", "major", "passing_year", "result"])
    experience = pd.DataFrame(experience, columns=[
        "resume_id", "company", "position", "start_date", "end_date"])

    matches = df[["resume_id", "job_id", "matched_score"]].copy()

    return {
        "resumes": resumes, "jobs": jobs, "skills": skills,
        "resume_skills": resume_skills, "job_skills": job_skills,
        "education": education, "experience": experience, "matches": matches,
    }


def create_database(csv_path=DEFAULT_CSV, db_path=DEFAULT_DB) -> dict[str, int]:
    """Create the SQLite database from scratch and return row counts per table."""
    tables = build_tables(load_resume_csv(csv_path))
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        for name, frame in tables.items():   # dict order respects foreign keys
            frame.to_sql(name, conn, if_exists="append", index=False)
        counts = {name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                  for name in tables}
    return counts


# ── Used by the web app and the notebooks ─────────────────────────────────────

def run_query(sql: str, db_path=DEFAULT_DB, params=()) -> pd.DataFrame:
    """Run a SQL query and return the result as a pandas DataFrame."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def save_analysis(job_key: str, result: dict, input_type: str = "pdf",
                  resume_chars: int | None = None, db_path=DEFAULT_DB) -> int | None:
    """Store one analysis result from the app. Returns the new analysis_id,
    or None if the database has not been built yet (the app keeps working)."""
    if not Path(db_path).exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT job_id FROM jobs WHERE job_key = ?", (job_key,)).fetchone()
            if row is None:
                return None
            cur = conn.execute(
                "INSERT INTO analysis_results (job_id, created_at, input_type, resume_chars,"
                " final_score, skill_overlap, tfidf_sim) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row[0], datetime.now().isoformat(timespec="seconds"), input_type, resume_chars,
                 result.get("final_score"), result.get("skill_overlap"), result.get("tfidf_sim")))
            analysis_id = cur.lastrowid
            for status, key in (("matched", "matched"), ("missing", "missing")):
                for skill in result.get(key, []):
                    name = normalize_skill(skill)
                    if not name:
                        continue
                    conn.execute("INSERT OR IGNORE INTO skills (name) VALUES (?)", (name,))
                    conn.execute(
                        "INSERT OR IGNORE INTO analysis_skills (analysis_id, skill_id, status) "
                        "SELECT ?, skill_id, ? FROM skills WHERE name = ?",
                        (analysis_id, status, name))
            return analysis_id
    except sqlite3.Error as exc:
        print(f"[database] could not save analysis: {exc}")
        return None


def export_results_json(path="results/analysis_results.json", db_path=DEFAULT_DB) -> int:
    """Export all stored analysis results to a JSON file. Returns the number of rows."""
    df = run_query("""
        SELECT a.analysis_id, a.created_at, j.title AS job, a.input_type,
               a.final_score, a.skill_overlap, a.tfidf_sim
        FROM analysis_results a JOIN jobs j ON j.job_id = a.job_id
        ORDER BY a.analysis_id""", db_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return len(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Resume Analyzer SQLite database")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to resume_data.csv")
    parser.add_argument("--db", default=DEFAULT_DB, help="output .db file")
    args = parser.parse_args()

    for table, n in create_database(args.csv, args.db).items():
        print(f"{table:<15} {n:>6} rows")
    print(f"\nDatabase saved to {args.db}")
