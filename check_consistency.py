"""
Item 1 check: do the website and the notebook use the same models and give the
same results?

    python check_consistency.py

It checks:
  1. library versions against the versions the saved files were created with;
  2. that tfidf.pkl is exactly what the notebook code produces (it refits the
     vectorizer on data/resume_data.csv with the notebook's settings);
  3. the notebook's example resume/vacancy run through nlp_pipeline with the
     unchanged saved model. Compare the printed numbers with the
     "РЕЗУЛЬТАТЫ АНАЛИЗА" cell of nlp-project.ipynb after re-running it.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nlp_pipeline as nlpp

BASE = Path(__file__).resolve().parent

# Same example as in nlp-project.ipynb
SAMPLE_RESUME = """
Experienced Data Scientist with 4 years of industry experience.
Proficient in Python, pandas, scikit-learn, TensorFlow, and PyTorch.
Strong background in machine learning, deep learning, and NLP.
Worked with PostgreSQL, MongoDB, AWS S3, and Spark.
Familiar with Docker, Git, and CI/CD pipelines.
Experience building REST APIs with FastAPI and Flask.
"""
SAMPLE_JOB = """
We are looking for a Machine Learning Engineer to join our team.
Requirements:
- Python (3+ years)
- TensorFlow or PyTorch
- Experience with NLP and transformer models
- SQL databases (PostgreSQL preferred)
- Docker and Kubernetes
- AWS or GCP cloud platforms
- MLflow or similar experiment tracking
- Strong understanding of statistics and algorithms
"""
EXPECTED_SKLEARN = "1.6.1"   # version tfidf.pkl was saved with


def warn(message: str) -> None:
    print("  WARN  " + message)


def check(ok: bool, message: str) -> bool:
    print(("  OK    " if ok else "  FAIL  ") + message)
    return ok


def check_versions() -> bool:
    print("\n1. Версии библиотек")
    meta = json.loads((BASE / "nlp_model" / "meta.json").read_text(encoding="utf-8"))
    need = meta["spacy_version"]                       # e.g. ">=3.8.0,<3.9.0"
    major_minor = ".".join(spacy.__version__.split(".")[:2])
    ok1 = check(major_minor == "3.8", f"spaCy {spacy.__version__} (модель требует {need})")
    ok2 = check(sklearn.__version__ == EXPECTED_SKLEARN,
                f"scikit-learn {sklearn.__version__} (tfidf.pkl сохранён в {EXPECTED_SKLEARN})")
    print(f"        Загружено приложением: {nlpp.MODEL_INFO}")
    return ok1 and ok2


def check_tfidf_reproducible() -> bool:
    print("\n2. tfidf.pkl совпадает с кодом ноутбука")
    df = pd.read_csv(BASE / "data" / "resume_data.csv")
    corpus = (df["career_objective"].fillna("") + " " + df["skills"].fillna("") + " "
              + df["responsibilities"].fillna("")).tolist()
    refit = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                            max_features=10000, sublinear_tf=True)
    cleaned = [nlpp.clean_text(t) for t in corpus]
    refit.fit(cleaned)
    with open(BASE / "tfidf.pkl", "rb") as f:
        saved = pickle.load(f)

    # When several words tie at the max_features=10000 cutoff, different
    # numpy/OS builds may keep different ones, so the word lists can differ
    # slightly. What matters: the same words get the same weights (idf), and
    # the example gets the same similarity score.
    shared = set(refit.vocabulary_) & set(saved.vocabulary_)
    share = len(shared) / len(saved.vocabulary_)
    idf_saved = {t: saved.idf_[i] for t, i in saved.vocabulary_.items()}
    idf_refit = {t: refit.idf_[i] for t, i in refit.vocabulary_.items()}
    idf_diff = max(abs(idf_saved[t] - idf_refit[t]) for t in shared)

    texts = [nlpp.clean_text(SAMPLE_RESUME), nlpp.clean_text(SAMPLE_JOB)]
    sim = lambda v: float(cosine_similarity(*v.transform(texts))[0][0])
    sim_saved, sim_refit = sim(saved), sim(refit)

    # How many words are tied exactly at the max_features cutoff?
    counts = CountVectorizer(ngram_range=(1, 2), stop_words="english").fit(cleaned)
    freq = np.asarray(counts.transform(cleaned).sum(axis=0)).ravel()
    limit = saved.max_features or len(freq)
    cutoff = np.sort(freq)[::-1][min(limit, len(freq)) - 1]
    tied = int((freq == cutoff).sum())
    tied_kept = limit - int((freq > cutoff).sum())
    arbitrary = saved.max_features is not None and tied > tied_kept

    ok_idf = check(idf_diff < 1e-9, f"веса idf общих терминов совпадают (макс. разница {idf_diff:.1e}) "
                                    "-> данные и настройки те же")
    if arbitrary:
        warn(f"{tied} терминов имеют одинаковую частоту {cutoff} на границе max_features={limit}; "
             f"из них случайно выбираются {tied_kept}")
        warn(f"поэтому словарь при переобучении на этом компьютере совпадает на {share:.1%}, "
             f"а сходство для примера: tfidf.pkl = {sim_saved:.4f}, переобучение = {sim_refit:.4f}")
        warn("причина: каждое резюме повторяется в CSV ~28 раз. Исправление в ноутбуке: обучать TF-IDF "
             "на уникальных резюме и убрать max_features (см. README)")
        print("        Приложение использует сам файл tfidf.pkl, поэтому сайт и Streamlit дают одинаковые "
              "результаты на любом компьютере.")
        return ok_idf
    ok_vocab = check(share >= 0.99, f"общих терминов {share:.2%} из {len(saved.vocabulary_)}")
    ok_sim = check(abs(sim_saved - sim_refit) < 1e-3,
                   f"сходство для примера: tfidf.pkl = {sim_saved:.4f}, код ноутбука = {sim_refit:.4f}")
    return ok_idf and ok_vocab and ok_sim


def check_example() -> None:
    print("\n3. Пример из ноутбука через nlp_pipeline (модель без добавленных навыков)")
    notebook_model, source = nlpp.load_spacy_model(extend=False)
    r = nlpp.analyze_texts(SAMPLE_RESUME, SAMPLE_JOB, model=notebook_model)
    print(f"        Модель: {source}")
    print(f"        Навыки кандидата ({len(r['resume_skills'])}): {', '.join(r['resume_skills'])}")
    print(f"        Требования вакансии ({len(r['job_skills'])}): {', '.join(r['job_skills'])}")
    print(f"        Skill Overlap     : {r['skill_overlap']:.1%}")
    print(f"        TF-IDF Similarity : {r['tfidf_sim']:.4f}")
    print(f"        Итоговая оценка   : {r['final_score']:.1%}")
    print("        -> Эти числа должны совпасть с ячейкой «ПРИМЕР РАБОТЫ СИСТЕМЫ»\n"
          "           в nlp-project.ipynb после перезапуска ноутбука (Run All).")


if __name__ == "__main__":
    results = [check_versions(), check_tfidf_reproducible()]
    check_example()
    verdict = "всё совпадает" if all(results) else "есть расхождения (см. FAIL выше)"
    print("\nИтог:", verdict + (" (есть предупреждения WARN)" if all(results) else ""))
    sys.exit(0 if all(results) else 1)
