import re
import ast
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Нормализация ──────────────────────────────────────────────────────────────

TECH_NORMALIZATIONS = {
    'js': 'javascript', 'py': 'python', 'ml': 'machine learning',
    'dl': 'deep learning', 'nlp': 'natural language processing',
    'tf': 'tensorflow', 'k8s': 'kubernetes', 'db': 'database',
    'ci': 'ci/cd', 'oop': 'object oriented programming',
}

def normalize_skill(skill: str) -> str:
    s = skill.lower().strip()
    return TECH_NORMALIZATIONS.get(s, s)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Словарь навыков ───────────────────────────────────────────────────────────

SKILLS_VOCAB = [
    # Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "golang", "rust",
    "kotlin", "swift", "scala", "r", "php", "ruby", "bash", "sql", "html", "css",
    "sass", "core java",
    # ML / AI
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data science", "data analysis", "data engineering",
    "scikit-learn", "tensorflow", "pytorch", "keras", "hugging face",
    "transformers", "bert", "gpt", "langchain", "openai", "llm",
    "reinforcement learning", "feature engineering", "model deployment",
    "mlflow", "xgboost", "lightgbm", "catboost", "random forest",
    "neural network", "convolutional neural network", "recurrent neural network",
    "risk prediction", "big data",
    # Data
    "data visualization", "data analytics", "business analytics", "business analysis",
    "data warehousing", "business intelligence", "tableau", "power bi",
    "apache spark", "spark", "pyspark", "hadoop", "hive", "kafka", "airflow",
    "dbt", "etl", "pandas", "numpy", "matplotlib", "seaborn",
    "amazon redshift", "google bigquery", "azure synapse", "snowflake",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "ci/cd", "github actions", "jenkins", "linux", "nginx", "redis",
    "cloud computing", "microservices", "serverless",
    # Databases
    "postgresql", "mongodb", "mysql", "sqlite", "elasticsearch", "cassandra",
    "dynamodb", "firebase", "nosql",
    # Backend
    "fastapi", "flask", "django", "spring boot", "node.js", "express.js",
    "rest api", "graphql", "grpc", "websocket", "celery",
    # Frontend
    "react", "react.js", "vue.js", "angular", "next.js", "nuxt.js",
    "tailwind", "webpack", "vite", "redux", "typescript",
    # Tools
    "git", "github", "gitlab", "jira", "confluence", "figma",
    "agile", "scrum", "kanban",
    # Other
    "application support", "network security", "system administration",
    "unity", "unreal engine", "opencv", "selenium", "pytest",
]

# ── spaCy pipeline (загружается один раз) ────────────────────────────────────

def build_nlp():
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    patterns = []
    for skill in SKILLS_VOCAB:
        skill_clean = skill.strip()
        if len(skill_clean) < 2:
            continue
        tokens = skill_clean.lower().split()
        if len(tokens) == 1:
            patterns.append({"label": "SKILL", "pattern": [{"LOWER": tokens[0]}]})
        else:
            patterns.append({
                "label": "SKILL",
                "pattern": [{"LOWER": t} for t in tokens]
            })
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    ruler.add_patterns(patterns)
    return nlp

nlp = build_nlp()

# TF-IDF vectorizer (обучается на словаре навыков как корпусе)
_tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)
_tfidf.fit(SKILLS_VOCAB + [" ".join(SKILLS_VOCAB)])

# ── Извлечение навыков ────────────────────────────────────────────────────────

def extract_skills(text: str) -> list[str]:
    text = clean_text(text)
    doc = nlp(text)
    skills = list(set([
        normalize_skill(ent.text)
        for ent in doc.ents
        if ent.label_ == "SKILL"
    ]))
    return sorted(skills)

# ── Matching ──────────────────────────────────────────────────────────────────

def skill_overlap(resume_skills: list, job_skills: list) -> dict:
    rs = set(resume_skills)
    js = set(job_skills)
    if not js:
        return {"score": 0.0, "matched": [], "missing": []}
    matched = sorted(rs & js)
    missing = sorted(js - rs)
    return {
        "score": round(len(matched) / len(js), 3),
        "matched": matched,
        "missing": missing,
    }

def tfidf_similarity(text_a: str, text_b: str) -> float:
    vecs = _tfidf.transform([text_a, text_b])
    return round(float(cosine_similarity(vecs[0], vecs[1])[0][0]), 4)

# ── Рекомендации ──────────────────────────────────────────────────────────────

LEARNING_TIPS = {
    "kubernetes":   "Пройдите курс CKA на killer.sh",
    "mlflow":       "Изучите MLflow через официальную документацию",
    "spark":        "Databricks Academy — бесплатный курс по Spark",
    "airflow":      "Настройте Airflow локально и создайте первые DAGs",
    "terraform":    "HashiCorp Learn — бесплатный курс Terraform",
    "golang":       "Изучите Go через tour.golang.org",
    "rust":         "Книга 'The Rust Programming Language' на doc.rust-lang.org",
    "aws":          "AWS Free Tier + курс AWS Cloud Practitioner",
    "docker":       "Play with Docker — бесплатная песочница",
    "pytorch":      "Официальные туториалы на pytorch.org",
    "tensorflow":   "TensorFlow Developer Certificate на coursera",
    "typescript":   "Официальная документация typescriptlang.org",
    "postgresql":   "PostgreSQL Tutorial на postgresqltutorial.com",
}

def generate_recommendations(overlap: dict, score: float) -> dict:
    if score >= 0.65:
        level, emoji = "Отличное соответствие", "🟢"
    elif score >= 0.40:
        level, emoji = "Хорошее соответствие", "🟡"
    else:
        level, emoji = "Низкое соответствие", "🔴"

    tips = []
    for skill in overlap["missing"][:5]:
        tip = LEARNING_TIPS.get(skill.lower(), f"Изучите {skill} через официальную документацию")
        tips.append({"skill": skill, "tip": tip})

    return {
        "level": level,
        "emoji": emoji,
        "tips": tips,
        "resume_advice": [
            "Добавьте конкретные метрики к достижениям (например, 'ускорил на 40%')",
            "Выделите раздел Technical Skills с явным перечислением технологий",
            "Адаптируйте ключевые слова резюме под конкретную вакансию",
        ]
    }

# ── Главная функция ───────────────────────────────────────────────────────────

def analyze(resume_text: str, job_skills: list[str], job_description: str) -> dict:
    resume_skills = extract_skills(resume_text)
    overlap       = skill_overlap(resume_skills, job_skills)
    tfidf         = tfidf_similarity(resume_text, job_description)
    final_score   = round(0.6 * overlap["score"] + 0.4 * tfidf, 3)
    recs          = generate_recommendations(overlap, final_score)

    return {
        "resume_skills":  resume_skills,
        "final_score":    final_score,
        "skill_overlap":  overlap["score"],
        "tfidf_sim":      tfidf,
        "matched":        overlap["matched"],
        "missing":        overlap["missing"],
        "recommendations": recs,
    }
