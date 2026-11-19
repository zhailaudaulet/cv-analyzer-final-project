"""
NLP pipeline of the Resume Analyzer: skill extraction (spaCy EntityRuler) and
resume–vacancy matching (skill overlap + TF-IDF cosine similarity).

Artifacts produced by nlp-project.ipynb (section 15):
    nlp_model/            spacy.blank("en") + EntityRuler; vocabulary = "vocab" split of the dataset
                          + the curated SKILLS_VOCAB below (with alias patterns from TECH_NORMALIZATIONS)
    tfidf.pkl             TF-IDF vectorizer fitted on the "train" split of the resumes (TFIDF_PARAMS)
    model_metadata.json   split sizes, seed, library versions, product weights/thresholds and the
                          held-out test metrics of exactly these artifacts (loaded into MODEL_INFO["metadata"])

The website, the Streamlit app and the notebook use the same artifacts. If a file is
missing or incompatible, the pipeline builds the same kind of model from SKILLS_VOCAB
and says so in MODEL_INFO.
"""
import json
import os
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "nlp_model"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

# ── Нормализация (та же, что в ноутбуке, раздел 1) ───────────────────────────

TECH_NORMALIZATIONS = {
    'js': 'javascript', 'py': 'python', 'ml': 'machine learning',
    'dl': 'deep learning', 'nlp': 'natural language processing',
    'tf': 'tensorflow', 'k8s': 'kubernetes', 'db': 'database',
    'oop': 'object oriented programming',
    'react.js': 'react', 'reactjs': 'react',
}

ALIASES = defaultdict(set)                       # нормализованный навык -> написания, которые к нему приводятся
for _alias, _canon in TECH_NORMALIZATIONS.items():
    ALIASES[_canon].add(_alias)


def normalize_skill(skill: str) -> str:
    s = skill.lower().strip()
    return TECH_NORMALIZATIONS.get(s, s)


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── Кураторский словарь навыков (EXTRA_SKILLS в ноутбуке, раздел 5) ──────────

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
    "react", "vue.js", "angular", "next.js", "nuxt.js",
    "tailwind", "webpack", "vite", "redux",
    # Tools
    "git", "github", "gitlab", "jira", "confluence", "figma",
    "agile", "scrum", "kanban",
    # Other
    "application support", "network security", "system administration",
    "unity", "unreal engine", "opencv", "selenium", "pytest",
]

TFIDF_PARAMS = dict(ngram_range=(1, 2), stop_words='english', max_features=10000, sublinear_tf=True)

# ── Загрузка моделей ─────────────────────────────────────────────────────────

MODEL_INFO = {"spacy": spacy.__version__, "sklearn": sklearn.__version__}


def skill_patterns(nlp, skills) -> list:
    """Паттерны EntityRuler для нормализованных форм и их алиасов, построенные токенизатором модели
    (так 'ci/cd', 'node.js', 'c++' разбиваются так же, как текст резюме)."""
    forms = set()
    for sk in skills:
        sk = normalize_skill(sk)
        forms.add(sk)
        forms.update(ALIASES.get(sk, ()))
    seen, patterns = set(), []
    for sk in sorted(forms):
        if len(sk) < 2:
            continue
        toks = tuple(t.lower_ for t in nlp.make_doc(sk))
        if toks and toks not in seen:
            seen.add(toks)
            patterns.append({"label": "SKILL", "pattern": [{"LOWER": t} for t in toks]})
    return patterns


def load_spacy_model(extend: bool = True):
    """Загружает nlp_model/. С extend=True в EntityRuler добавляются навыки из SKILLS_VOCAB,
    которых в нём нет (для модели из текущего ноутбука их 0)."""
    try:
        nlp = spacy.load(MODEL_DIR)
        source = "nlp_model/ (модель из ноутбука)"
    except Exception as exc:                      # нет папки или несовместимая версия spaCy
        warnings.warn(f"Не удалось загрузить {MODEL_DIR.name}: {exc}. Используется spacy.blank('en') + SKILLS_VOCAB.")
        nlp = spacy.blank("en")
        nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
        source = "spacy.blank('en') + SKILLS_VOCAB (fallback)"
        extend = True

    if extend:
        ruler = nlp.get_pipe("entity_ruler")
        existing = {tuple(tok.get("LOWER") for tok in p["pattern"])
                    for p in ruler.patterns if isinstance(p["pattern"], list)}
        new = [p for p in skill_patterns(nlp, SKILLS_VOCAB)
               if tuple(tok["LOWER"] for tok in p["pattern"]) not in existing]
        if new:
            ruler.add_patterns(new)
            source += f", +{len(new)} навыков из SKILLS_VOCAB"
    return nlp, source


def load_vectorizer():
    """Загружает TF-IDF из tfidf.pkl (обучен в ноутбуке на части train)."""
    try:
        with open(TFIDF_PATH, "rb") as f:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                vectorizer = pickle.load(f)
        for w in caught:                          # например, другая версия sklearn
            print(f"[nlp_pipeline] {w.message}")
        return vectorizer, f"tfidf.pkl ({len(vectorizer.vocabulary_)} терминов)"
    except Exception as exc:
        warnings.warn(f"Не удалось загрузить {TFIDF_PATH.name}: {exc}. TF-IDF обучается на SKILLS_VOCAB.")
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS).fit(SKILLS_VOCAB + [" ".join(SKILLS_VOCAB)])
        return vectorizer, "SKILLS_VOCAB (fallback)"


def load_metadata():
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


nlp, MODEL_INFO["ner_model"] = load_spacy_model(extend=True)
_tfidf, MODEL_INFO["tfidf"] = load_vectorizer()
MODEL_INFO["metadata"] = load_metadata()

# Итоговый балл и уровни — из model_metadata.json, если он есть (значения по умолчанию те же, что в ноутбуке)
_product = (MODEL_INFO["metadata"] or {}).get("product_score", {})
W_OVERLAP = float(_product.get("w_overlap", 0.6))
LEVEL_THRESHOLDS = tuple(_product.get("level_thresholds", (0.40, 0.65)))
LEVELS = [("Низкое соответствие", "🔴"), ("Хорошее соответствие", "🟡"), ("Отличное соответствие", "🟢")]


def level_index(score: float) -> int:
    """0 — низкое, 1 — хорошее, 2 — отличное."""
    return int(np.digitize(score, LEVEL_THRESHOLDS))

# ── Извлечение навыков ────────────────────────────────────────────────────────

def extract_skills(text: str, model=None) -> list[str]:
    doc = (model if model is not None else nlp)(clean_text(text))
    return sorted({normalize_skill(ent.text) for ent in doc.ents if ent.label_ == "SKILL"})

# ── Matching ──────────────────────────────────────────────────────────────────

def skill_overlap(resume_skills: list, job_skills: list) -> dict:
    rs = {normalize_skill(s) for s in resume_skills}
    js = {normalize_skill(s) for s in job_skills}
    if not js:
        return {"score": 0.0, "matched": [], "missing": [], "extra": []}
    matched = rs & js
    return {"score": round(len(matched) / len(js), 3), "matched": sorted(matched),
            "missing": sorted(js - rs), "extra": sorted(rs - js)}


def tfidf_similarity(text_a: str, text_b: str) -> float:
    vecs = _tfidf.transform([clean_text(text_a), clean_text(text_b)])
    return round(float(vecs[0].multiply(vecs[1]).sum()), 4)     # строки L2-нормированы → скалярное произведение = cosine


def final_score(overlap_score: float, tfidf: float) -> float:
    return round(W_OVERLAP * overlap_score + (1 - W_OVERLAP) * tfidf, 4)

# ── Рекомендации ──────────────────────────────────────────────────────────────

LEARNING_TIPS = {
    "kubernetes": "пройдите курс CKA и практикуйтесь на killer.sh",
    "mlflow":     "изучите MLflow по официальной документации",
    "spark":      "Databricks Academy — бесплатный курс по Spark",
    "airflow":    "поднимите Airflow локально и напишите первые DAG",
    "terraform":  "HashiCorp Learn — бесплатный курс Terraform",
    "golang":     "tour.golang.org",
    "rust":       "книга 'The Rust Programming Language' на doc.rust-lang.org",
    "docker":     "Play with Docker — бесплатная песочница",
    "aws":        "AWS Free Tier + курс AWS Cloud Practitioner",
    "pytorch":    "официальные туториалы на pytorch.org",
    "tensorflow": "TensorFlow Developer Certificate",
    "typescript": "документация typescriptlang.org",
    "postgresql": "postgresqltutorial.com",
}
DEFAULT_TIP = "изучите по официальной документации и сделайте небольшой проект"
RESUME_ADVICE = [
    "Добавьте измеримые результаты (например, «ускорил обработку на 40%»).",
    "Вынесите раздел Technical Skills с явным списком технологий.",
    "Используйте формулировки навыков из текста вакансии.",
]


def generate_recommendations(overlap: dict, score: float) -> dict:
    """Структурированные рекомендации для сайта и Streamlit."""
    level, emoji = LEVELS[level_index(score)]
    tips = [{"skill": s, "tip": LEARNING_TIPS.get(s.lower(), DEFAULT_TIP)} for s in overlap["missing"][:5]]
    return {"level": level, "emoji": emoji, "tips": tips, "resume_advice": list(RESUME_ADVICE)}


def build_prompt(resume_text: str, job_description: str, overlap: dict) -> str:
    matched = ', '.join(overlap['matched']) if overlap['matched'] else 'нет совпадений'
    missing = ', '.join(overlap['missing']) if overlap['missing'] else 'все навыки присутствуют'
    return f"""You are an expert career coach and technical recruiter specializing in IT.
Analyze the following resume against a job description and provide actionable recommendations.

## Resume:
{resume_text.strip()}

## Job Description:
{job_description.strip()}

## Pre-computed Analysis:
- Skill match score: {overlap['score']:.0%}
- Matched skills: {matched}
- Missing required skills: {missing}

## Task:
Provide a structured response with:
1. **Overall Assessment** (2-3 sentences on the candidate's fit)
2. **Top 3 Strengths** (skills/experience that align well)
3. **Top 3 Gaps** (most critical missing skills or experience)
4. **Action Plan** (specific steps to improve match rate, e.g., courses, projects)
5. **Resume Optimization Tips** (2-3 concrete changes to the resume text)

Be specific, concise, and actionable. Focus on IT/tech context."""


def get_groq_recommendations(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY не задан")
    from groq import Groq
    resp = Groq(api_key=key).chat.completions.create(messages=[{"role": "user", "content": prompt}], model=model)
    return resp.choices[0].message.content


def rule_based_recommendations(overlap: dict, score: float) -> str:
    level = LEVELS[level_index(score)][0].split()[0].lower()
    matched, missing = overlap["matched"], overlap["missing"]
    lines = ["## Overall Assessment",
             f"Итоговое соответствие — {level} ({score:.0%}); совпало {len(matched)} из {len(matched) + len(missing)} требуемых навыков.",
             "## Top 3 Strengths"] + [f"- {s}" for s in matched[:3]] + (["- (совпадений навыков нет)"] if not matched else [])
    lines += ["## Top 3 Gaps"] + [f"- {s}" for s in missing[:3]] + (["- (пробелов нет)"] if not missing else [])
    lines += ["## Action Plan"] + [f"- {s}: {LEARNING_TIPS.get(s, DEFAULT_TIP)}" for s in missing[:5]]
    lines += ["## Resume Optimization Tips"] + [f"- {a}" for a in RESUME_ADVICE]
    return "\n".join(lines)


def get_recommendations(prompt: str, overlap: dict, score: float, use_llm="auto"):
    """LLM (Groq), если доступен; иначе rule-based. Возвращает (текст, источник)."""
    if use_llm in ("auto", True):
        try:
            return get_groq_recommendations(prompt), "groq"
        except Exception:
            if use_llm is True:
                raise
    return rule_based_recommendations(overlap, score), "rule-based"

# ── Главные функции ───────────────────────────────────────────────────────────

def _match(resume_text: str, job_skills: list, job_text: str, model=None) -> dict:
    resume_skills = extract_skills(resume_text, model)
    overlap = skill_overlap(resume_skills, job_skills)
    tfidf = tfidf_similarity(resume_text, job_text)
    return {"resume_skills": resume_skills, "overlap": overlap, "tfidf": tfidf,
            "final_score": final_score(overlap["score"], tfidf)}


def analyze(resume_text: str, job_skills: list[str], job_description: str) -> dict:
    """Сайт и Streamlit: навыки вакансии берутся из jobs.py."""
    m = _match(resume_text, job_skills, job_description)
    return {
        "resume_skills":   m["resume_skills"],
        "final_score":     m["final_score"],
        "skill_overlap":   m["overlap"]["score"],
        "tfidf_sim":       m["tfidf"],
        "matched":         m["overlap"]["matched"],
        "missing":         m["overlap"]["missing"],
        "recommendations": generate_recommendations(m["overlap"], m["final_score"]),
    }


def analyze_texts(resume_text: str, job_text: str, model=None) -> dict:
    """Те же шаги, что analyze_resume() в ноутбуке: навыки обоих текстов извлекает модель.
    Используется check_consistency.py и ноутбуком (раздел 15) для сверки баллов."""
    job_skills = extract_skills(job_text, model)
    m = _match(resume_text, job_skills, job_text, model)
    return {
        "resume_skills": m["resume_skills"],
        "job_skills":    job_skills,
        "skill_overlap": m["overlap"]["score"],
        "tfidf_sim":     m["tfidf"],
        "final_score":   m["final_score"],
        "matched":       m["overlap"]["matched"],
        "missing":       m["overlap"]["missing"],
    }


def analyze_resume(resume_text: str, job_description: str, use_llm="auto") -> dict:
    """Порт analyze_resume() из ноутбука (раздел 14): балл + текстовые рекомендации (LLM или rule-based)."""
    job_skills = extract_skills(job_description)
    m = _match(resume_text, job_skills, job_description)
    prompt = build_prompt(resume_text, job_description, m["overlap"])
    recs, source = get_recommendations(prompt, m["overlap"], m["final_score"], use_llm)
    return {"resume_skills": m["resume_skills"], "job_skills": job_skills, "skill_overlap": m["overlap"],
            "tfidf_similarity": m["tfidf"], "final_score": m["final_score"], "llm_prompt": prompt,
            "recommendations": recs, "recommendations_source": source}
