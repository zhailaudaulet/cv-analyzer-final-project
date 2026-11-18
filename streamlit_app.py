"""
Streamlit dashboard for the Resume Analyzer (requirement 7: interactive UI).

Run:
    streamlit run streamlit_app.py

Pages (sidebar menu):
  1. Анализ резюме   — choose a vacancy, upload a PDF or paste text, get the score
  2. История         — analyses saved in SQLite, with filters, export to JSON/CSV
  3. SQL-аналитика   — run any of the 10 queries from sql_queries.py
  4. О модели        — which models and library versions are loaded
"""
import io
from pathlib import Path

import pandas as pd
import pypdf
import streamlit as st

from database import DEFAULT_DB, export_results_json, run_query, save_analysis
from jobs import JOBS
from sql_queries import QUERIES, run_named

st.set_page_config(page_title="Resume Analyzer", layout="wide")


@st.cache_resource(show_spinner="Загружаем NLP-модель…")
def load_pipeline():
    """Import once: loading the spaCy model takes a few seconds."""
    import nlp_pipeline
    return nlp_pipeline


def pdf_to_text(uploaded_file) -> str | None:
    """Text of an uploaded PDF, or None if it cannot be read."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        st.error(f"Не удалось прочитать PDF: {exc}")
        return None


def show_result(result: dict) -> None:
    rec = result["recommendations"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Итоговая оценка", f"{result['final_score']:.0%}", help="0.6 × навыки + 0.4 × TF-IDF")
    c2.metric("Совпадение навыков", f"{result['skill_overlap']:.0%}")
    c3.metric("Сходство текста (TF-IDF)", f"{result['tfidf_sim']:.0%}")
    st.subheader(f"{rec['emoji']} {rec['level']}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Совпадают с вакансией**")
        st.write(", ".join(result["matched"]) or "—")
        st.markdown("**Все навыки в резюме**")
        st.write(", ".join(result["resume_skills"]) or "—")
    with right:
        st.markdown("**Отсутствуют**")
        st.write(", ".join(result["missing"]) or "—")
        if rec["tips"]:
            st.markdown("**Что изучить**")
            for tip in rec["tips"]:
                st.write(f"- **{tip['skill']}**: {tip['tip']}")
    with st.expander("Советы по резюме"):
        for advice in rec["resume_advice"]:
            st.write(f"- {advice}")


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_analyze():
    st.title("Анализ резюме")
    titles = {job.title: key for key, job in JOBS.items()}
    job_key = titles[st.selectbox("Вакансия", list(titles))]
    job = JOBS[job_key]

    with st.expander("Описание вакансии"):
        st.write(job.description)
        st.markdown("**Требования:** " + "; ".join(job.requirements))
        st.markdown("**Навыки для сравнения:** " + ", ".join(job.skills))

    tab_pdf, tab_text = st.tabs(["PDF файл", "Вставить текст"])
    with tab_pdf:
        uploaded = st.file_uploader("Резюме в PDF", type=["pdf"])
        run_pdf = st.button("Анализировать PDF", disabled=uploaded is None)
    with tab_text:
        pasted = st.text_area("Текст резюме (на английском)", height=220)
        run_text = st.button("Анализировать текст")
        if run_text and len(pasted.strip()) < 50:
            st.error("Слишком короткий текст: нужно минимум 50 символов.")
            run_text = False

    resume_text, input_type = None, None
    if run_pdf and uploaded is not None:
        resume_text, input_type = pdf_to_text(uploaded), "pdf"
        if resume_text is not None and len(resume_text.strip()) < 50:
            st.error("В PDF нет текста (возможно, это скан).")
            resume_text = None
    elif run_text:
        resume_text, input_type = pasted, "text"

    if resume_text:
        nlp_pipeline = load_pipeline()
        with st.spinner("Анализируем…"):
            result = nlp_pipeline.analyze(resume_text, job.skills, job.text_en)
        analysis_id = save_analysis(job_key, result, input_type, len(resume_text))
        show_result(result)
        if analysis_id:
            st.caption(f"Результат сохранён в базу (analysis_id = {analysis_id}).")
        else:
            st.caption("База данных не найдена — запустите `python database.py`, чтобы сохранять историю.")


def page_history():
    st.title("История анализов")
    if not Path(DEFAULT_DB).exists():
        st.warning("База данных не создана. Выполните в терминале: `python database.py`")
        return
    df = run_query("""
        SELECT a.analysis_id, a.created_at, j.title AS vacancy, a.input_type,
               a.final_score, a.skill_overlap, a.tfidf_sim
        FROM analysis_results a JOIN jobs j ON j.job_id = a.job_id
        ORDER BY a.analysis_id DESC""")
    if df.empty:
        st.info("Пока нет ни одного анализа. Проверьте резюме на странице «Анализ резюме».")
        return

    c1, c2 = st.columns(2)
    vacancies = c1.multiselect("Вакансия", sorted(df["vacancy"].unique()))
    min_score = c2.slider("Минимальная итоговая оценка", 0.0, 1.0, 0.0, 0.05)
    view = df[df["final_score"] >= min_score]
    if vacancies:
        view = view[view["vacancy"].isin(vacancies)]

    st.dataframe(view, use_container_width=True, hide_index=True)
    st.bar_chart(view.groupby("vacancy")["final_score"].mean(), y_label="Средняя оценка")

    c1, c2 = st.columns(2)
    c1.download_button("Скачать CSV", view.to_csv(index=False).encode("utf-8"),
                       "analysis_results.csv", "text/csv")
    if c2.button("Сохранить всё в results/analysis_results.json"):
        n = export_results_json()
        st.success(f"Сохранено записей: {n}")


def page_sql():
    st.title("SQL-аналитика")
    if not Path(DEFAULT_DB).exists():
        st.warning("База данных не создана. Выполните в терминале: `python database.py`")
        return
    names = list(QUERIES)
    name = st.selectbox("Запрос", names,
                        format_func=lambda n: f"{names.index(n) + 1}. {QUERIES[n]['title']}")
    q = QUERIES[name]
    st.write(f"**Вопрос:** {q['question']}")
    st.write(f"**Техники SQL:** {', '.join(q['techniques'])}")
    with st.expander("Показать SQL"):
        st.code(q["sql"].strip(), language="sql")
    try:
        st.dataframe(run_named(name), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Ошибка запроса: {exc}")


def page_model():
    st.title("О модели")
    nlp_pipeline = load_pipeline()
    st.json(nlp_pipeline.MODEL_INFO)
    st.markdown(
        "- **Извлечение навыков:** spaCy EntityRuler (`nlp_model/`, обучен в `nlp-project.ipynb`)\n"
        "- **Совпадение навыков:** доля навыков вакансии, найденных в резюме\n"
        "- **Сходство текста:** TF-IDF (`tfidf.pkl`) + косинусное сходство\n"
        "- **Итоговая оценка:** 0.6 × совпадение навыков + 0.4 × TF-IDF"
    )


PAGES = {
    "Анализ резюме": page_analyze,
    "История": page_history,
    "SQL-аналитика": page_sql,
    "О модели": page_model,
}
choice = st.sidebar.radio("Меню", list(PAGES))
PAGES[choice]()
