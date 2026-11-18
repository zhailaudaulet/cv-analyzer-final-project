-- Resume Analyzer — SQLite schema (3NF)
-- Source data: data/resume_data.csv (344 resumes × 28 job postings, 9 544 resume–job pairs)
-- Built by: python database.py

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS analysis_skills;
DROP TABLE IF EXISTS analysis_results;
DROP TABLE IF EXISTS matches;
DROP TABLE IF EXISTS job_skills;
DROP TABLE IF EXISTS resume_skills;
DROP TABLE IF EXISTS experience;
DROP TABLE IF EXISTS education;
DROP TABLE IF EXISTS skills;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS resumes;

-- One row per unique candidate resume
CREATE TABLE resumes (
    resume_id         INTEGER PRIMARY KEY,
    address           TEXT,
    career_objective  TEXT,
    languages         TEXT,
    skills_raw        TEXT            -- original list string, kept for auditing
);

-- One row per job posting. source = 'dataset' (28 postings from the CSV)
-- or 'app' (the 5 vacancies hard-coded in jobs.py)
CREATE TABLE jobs (
    job_id                   INTEGER PRIMARY KEY,
    job_key                  TEXT UNIQUE,     -- jobs.py id for app vacancies, NULL for dataset jobs
    source                   TEXT NOT NULL CHECK (source IN ('dataset', 'app')),
    title                    TEXT NOT NULL,
    education_requirement    TEXT,
    experience_requirement   TEXT,
    min_experience_years     INTEGER,         -- parsed from experience_requirement, NULL if not stated
    age_requirement          TEXT,
    responsibilities         TEXT
);

-- Skill dictionary: every normalized skill name appears once
CREATE TABLE skills (
    skill_id  INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);

-- Many-to-many: which skills each resume lists
CREATE TABLE resume_skills (
    resume_id  INTEGER NOT NULL REFERENCES resumes(resume_id),
    skill_id   INTEGER NOT NULL REFERENCES skills(skill_id),
    PRIMARY KEY (resume_id, skill_id)
);

-- Many-to-many: which skills each job requires
CREATE TABLE job_skills (
    job_id    INTEGER NOT NULL REFERENCES jobs(job_id),
    skill_id  INTEGER NOT NULL REFERENCES skills(skill_id),
    PRIMARY KEY (job_id, skill_id)
);

-- Education entries of a resume (one row per degree)
CREATE TABLE education (
    education_id   INTEGER PRIMARY KEY,
    resume_id      INTEGER NOT NULL REFERENCES resumes(resume_id),
    institution    TEXT,
    degree         TEXT,
    major          TEXT,
    passing_year   TEXT,
    result         TEXT
);

-- Work experience entries of a resume (one row per position)
CREATE TABLE experience (
    experience_id  INTEGER PRIMARY KEY,
    resume_id      INTEGER NOT NULL REFERENCES resumes(resume_id),
    company        TEXT,
    position       TEXT,
    start_date     TEXT,
    end_date       TEXT
);

-- Ground-truth labels from the dataset: match score for each resume–job pair
CREATE TABLE matches (
    resume_id      INTEGER NOT NULL REFERENCES resumes(resume_id),
    job_id         INTEGER NOT NULL REFERENCES jobs(job_id),
    matched_score  REAL NOT NULL CHECK (matched_score BETWEEN 0 AND 1),
    PRIMARY KEY (resume_id, job_id)
);

-- Results produced by the application (one row per analysis run)
CREATE TABLE analysis_results (
    analysis_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id         INTEGER NOT NULL REFERENCES jobs(job_id),
    created_at     TEXT NOT NULL,             -- ISO 8601 timestamp
    input_type     TEXT NOT NULL CHECK (input_type IN ('pdf', 'text')),
    resume_chars   INTEGER,
    final_score    REAL,
    skill_overlap  REAL,
    tfidf_sim      REAL
);

-- Skills found / missing in each analysis run
CREATE TABLE analysis_skills (
    analysis_id  INTEGER NOT NULL REFERENCES analysis_results(analysis_id),
    skill_id     INTEGER NOT NULL REFERENCES skills(skill_id),
    status       TEXT NOT NULL CHECK (status IN ('matched', 'missing', 'extra')),
    PRIMARY KEY (analysis_id, skill_id)
);

CREATE INDEX idx_resume_skills_skill ON resume_skills(skill_id);
CREATE INDEX idx_job_skills_skill    ON job_skills(skill_id);
CREATE INDEX idx_matches_job         ON matches(job_id);
CREATE INDEX idx_analysis_job        ON analysis_results(job_id);
