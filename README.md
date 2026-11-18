# Resume Analyzer

Система автоматического извлечения навыков из резюме и оценки соответствия резюме IT-вакансиям.

**Курс:** Programming for Data Analysis and Database (ADA-2603M) · **Тема 16:** CV/Resume Analyzer
**Команда:** Daulet Zhailau · Asset Kapi · Akmaral Amanzholova

## Что умеет проект

- извлекает навыки из резюме (spaCy EntityRuler, ~3 900 шаблонов навыков);
- сравнивает резюме с вакансией: совпадение навыков + TF-IDF сходство текстов;
- показывает отсутствующие навыки и рекомендации, что изучить;
- принимает резюме в PDF **или** как вставленный текст;
- хранит данные и результаты анализа в SQLite, содержит 10 аналитических SQL-запросов;
- два интерфейса: веб-сайт (FastAPI) и дашборд (Streamlit).

## Данные

[Resume Dataset (Kaggle, saugataroyarghya)](https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset) → файл `data/resume_data.csv` (17 МБ, 9 544 строки, 35 колонок).

Важно: строки — это **пары «резюме–вакансия»**: 344 уникальных резюме × 28 вакансий с оценкой соответствия `matched_score`. Подробный аудит данных — в `data_prep.ipynb`.

Если файла нет в `data/`, скачайте его со страницы Kaggle (кнопка Download) и положите в папку `data/`.

## Структура

```
cv-analyzer-final-project/
├── main.py                 # FastAPI: страницы сайта и API (PDF и текст)
├── streamlit_app.py        # Streamlit-дашборд: анализ, история, SQL-аналитика
├── nlp_pipeline.py         # Извлечение навыков + matching; загружает nlp_model/ и tfidf.pkl
├── database.py             # Создание SQLite-базы из CSV, сохранение результатов
├── sql_queries.py          # 10 аналитических SQL-запросов
├── jobs.py                 # 5 вакансий для сайта
├── check_consistency.py    # Проверка: сайт и ноутбук дают одинаковые результаты
├── nlp-project.ipynb       # Модели: NER, TF-IDF, оценка, рекомендации LLM
├── data_prep.ipynb         # Качество данных, память, chunksize, CSV vs Parquet
├── sql_analysis.ipynb      # SQL-запросы → Pandas, графики, SQL vs Pandas
├── nlp_model/              # Сохранённая spaCy-модель (из nlp-project.ipynb)
├── tfidf.pkl               # Сохранённый TF-IDF (из nlp-project.ipynb)
├── db/schema.sql           # Схема базы данных
├── docs/er_diagram.png     # ER-диаграмма
├── data/resume_data.csv    # Датасет
├── static/                 # HTML/CSS/JS сайта
└── requirements.txt
```

## Установка

Нужен **Python 3.11** (подходит и 3.12; на 3.13 зависимости не устанавливаются).

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## Запуск

```bash
python database.py                      # 1. создать базу resume_analyzer.db (≈ 2 секунды)
python -m uvicorn main:app --reload     # 2a. сайт: http://127.0.0.1:8000
streamlit run streamlit_app.py          # 2b. или дашборд: http://localhost:8501
```

## Как считается оценка

1. Текст резюме очищается (`clean_text`).
2. spaCy EntityRuler находит навыки; названия нормализуются (`js` → `javascript`, `react.js` → `react`).
3. **Совпадение навыков** = доля навыков вакансии, найденных в резюме.
4. **TF-IDF сходство** = косинусное сходство векторов резюме и английского текста вакансии.
5. **Итоговая оценка** = 0.6 × совпадение навыков + 0.4 × TF-IDF.

## База данных

10 таблиц в 3НФ (схема — `db/schema.sql`, диаграмма — `docs/er_diagram.png`): `resumes`, `jobs`, `skills`, `resume_skills`, `job_skills`, `education`, `experience`, `matches` (истинные оценки), `analysis_results` и `analysis_skills` (результаты из приложения).

## Версии

Python 3.11 · spaCy 3.8 · scikit-learn 1.6.1 · NumPy 2.2 · pandas 2.2 · FastAPI 0.115 · Streamlit 1.40 — полный список в `requirements.txt`.
`nlp_model/` сохранена в spaCy 3.8, `tfidf.pkl` — в scikit-learn 1.6.1; другие версии могут не загрузить эти файлы (тогда приложение переходит на запасной вариант и пишет об этом).

## Ограничения

- Объём датасета (17 МБ, 344 резюме) меньше рекомендуемых 200 МБ .
- Шаблоны навыков построены из самого датасета, поэтому новые технологии могут не распознаваться.
- Резюме должно быть на английском; сканированные PDF без текстового слоя не поддерживаются.
