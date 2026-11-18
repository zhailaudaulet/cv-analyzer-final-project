# Resume Analyzer

Система автоматического извлечения навыков и оценки соответствия резюме вакансиям в IT.

## Запуск

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
uvicorn main:app --reload
```

Открыть: http://localhost:8000

## Структура

```
resume-analyzer/
├── main.py            # FastAPI — роуты и PDF-парсинг
├── nlp_pipeline.py    # NER (spaCy EntityRuler) + TF-IDF matching
├── jobs.py            # Описания вакансий
├── static/
│   ├── index.html     # Главная страница — список вакансий
│   └── job.html       # Страница вакансии — загрузка PDF + результаты
└── requirements.txt
```

## Как работает

1. Пользователь выбирает вакансию на главной странице
2. Загружает PDF резюме
3. FastAPI парсит PDF через `pypdf`
4. `nlp_pipeline.analyze()` запускает:
   - **NER** (spaCy EntityRuler) — извлекает навыки из текста
   - **Skill Overlap** — пересечение навыков резюме и вакансии
   - **TF-IDF Cosine Similarity** — семантическое сходство текстов
   - Итоговая оценка = 0.6 × Skill Overlap + 0.4 × TF-IDF
5. Фронтенд отображает score, совпавшие/отсутствующие навыки и рекомендации
