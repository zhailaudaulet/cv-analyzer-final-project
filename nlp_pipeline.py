"""
NLP pipeline of the Resume Analyzer: skill extraction (spaCy EntityRuler) and
resume–vacancy matching (skill overlap + TF-IDF cosine similarity).

The trained artifacts come from nlp-project.ipynb:
    nlp_model/   spaCy pipeline with an EntityRuler of ~3 900 skill patterns
    tfidf.pkl    TF-IDF vectorizer fitted on the resume corpus

Both are loaded here, so the website, the Streamlit app and the notebook use the
same models. If a file is missing or was saved with incompatible library
versions, the pipeline falls back to en_core_web_md + the built-in vocabulary
below and says so in MODEL_INFO.
"""
import pickle
import re
import warnings
from pathlib import Path

import sklearn
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "nlp_model"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"

# ── Нормализация (та же, что в ноутбуке) ─────────────────────────────────────

TECH_NORMALIZATIONS = {
    'js': 'javascript', 'py': 'python', 'ml': 'machine learning',
    'dl': 'deep learning', 'nlp': 'natural language processing',
    'tf': 'tensorflow', 'k8s': 'kubernetes', 'db': 'database',
    'oop': 'object oriented programming',
    # дубликаты одного навыка в разных написаниях
    'react.js': 'react', 'reactjs': 'react',
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

# ── Словарь навыков вакансий ─────────────────────────────────────────────────
# Добавляется к сохранённой модели, чтобы все навыки из jobs.py распознавались
# (в модели из ноутбука нет, например, mlflow, nginx, terraform).

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

# ── Загрузка моделей ─────────────────────────────────────────────────────────

MODEL_INFO = {"spacy": spacy.__version__, "sklearn": sklearn.__version__}


def _skill_patterns(nlp, skills):
    """Token patterns built with the model's own tokenizer, so 'ci/cd',
    'node.js' or 'c++' are split exactly like the resume text will be."""
    patterns = []
    for skill in skills:
        tokens = [t.lower_ for t in nlp.make_doc(skill.strip())]
        if tokens and len(skill.strip()) >= 2:
            patterns.append({"label": "SKILL", "pattern": [{"LOWER": t} for t in tokens]})
    return patterns


def load_spacy_model(extend: bool = True):
    """Load the saved pipeline from nlp_model/. With extend=True the vacancy
    skills from SKILLS_VOCAB are added to its EntityRuler."""
    try:
        nlp = spacy.load(MODEL_DIR)
        source = "nlp_model/ (сохранённая модель из ноутбука)"
    except Exception as exc:                      # нет папки или несовместимая версия
        warnings.warn(f"Не удалось загрузить {MODEL_DIR.name}: {exc}. "
                      "Используется en_core_web_md + встроенный словарь.")
        nlp = spacy.load("en_core_web_md", disable=["ner"])
        nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        source = "en_core_web_md + встроенный словарь (fallback)"
        extend = True

    if extend:
        ruler = nlp.get_pipe("entity_ruler")
        existing = {tuple(tok.get("LOWER") for tok in p["pattern"])
                    for p in ruler.patterns if isinstance(p["pattern"], list)}
        new = [p for p in _skill_patterns(nlp, SKILLS_VOCAB)
               if tuple(tok["LOWER"] for tok in p["pattern"]) not in existing]
        ruler.add_patterns(new)
        source += f", +{len(new)} навыков вакансий"
    return nlp, source


def load_vectorizer():
    """Load the TF-IDF vectorizer fitted in the notebook (tfidf.pkl)."""
    try:
        with open(TFIDF_PATH, "rb") as f:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                vectorizer = pickle.load(f)
        for w in caught:                          # например, другая версия sklearn
            print(f"[nlp_pipeline] {w.message}")
        return vectorizer, f"tfidf.pkl ({len(vectorizer.vocabulary_)} терминов)"
    except Exception as exc:
        warnings.warn(f"Не удалось загрузить {TFIDF_PATH.name}: {exc}. "
                      "TF-IDF обучается на встроенном словаре.")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        vectorizer.fit(SKILLS_VOCAB + [" ".join(SKILLS_VOCAB)])
        return vectorizer, "встроенный словарь (fallback)"


nlp, MODEL_INFO["ner_model"] = load_spacy_model(extend=True)
_tfidf, MODEL_INFO["tfidf"] = load_vectorizer()

# ── Извлечение навыков ────────────────────────────────────────────────────────

def extract_skills(text: str, model=None) -> list[str]:
    doc = (model if model is not None else nlp)(clean_text(text))
    return sorted({normalize_skill(ent.text) for ent in doc.ents if ent.label_ == "SKILL"})

# ── Matching ──────────────────────────────────────────────────────────────────

def skill_overlap(resume_skills: list, job_skills: list) -> dict:
    rs = {normalize_skill(s) for s in resume_skills}
    js = {normalize_skill(s) for s in job_skills}
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
    vecs = _tfidf.transform([clean_text(text_a), clean_text(text_b)])
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

# ── Главные функции ───────────────────────────────────────────────────────────

def analyze(resume_text: str, job_skills: list[str], job_description: str) -> dict:
    """Used by the website and the Streamlit app: the vacancy's skills come from jobs.py."""
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


def analyze_texts(resume_text: str, job_text: str, model=None) -> dict:
    """Same steps as analyze_resume() in nlp-project.ipynb (skills of BOTH texts are
    extracted by the model). Used by check_consistency.py to compare with the notebook."""
    resume_skills = extract_skills(resume_text, model)
    job_skills    = extract_skills(job_text, model)
    overlap       = skill_overlap(resume_skills, job_skills)
    tfidf         = tfidf_similarity(resume_text, job_text)
    return {
        "resume_skills": resume_skills,
        "job_skills":    job_skills,
        "skill_overlap": overlap["score"],
        "tfidf_sim":     tfidf,
        "final_score":   round(0.6 * overlap["score"] + 0.4 * tfidf, 4),
        "matched":       overlap["matched"],
        "missing":       overlap["missing"],
    }
