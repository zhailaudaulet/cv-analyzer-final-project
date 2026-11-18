import io

import pypdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import save_analysis
from jobs import JOBS
from nlp_pipeline import MODEL_INFO, analyze

app = FastAPI(title="Resume Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")

MAX_PDF_BYTES = 5 * 1024 * 1024      # 5 МБ
MIN_TEXT_CHARS = 50
MAX_TEXT_CHARS = 50_000


class ResumeText(BaseModel):
    text: str

# ── Страницы ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_page(job_id: str):
    get_job_or_404(job_id)
    with open("static/job.html", encoding="utf-8") as f:
        return f.read()

# ── Вспомогательные функции ───────────────────────────────────────────────────

def get_job_or_404(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


def pdf_to_text(content: bytes) -> str:
    """Извлекает текст из PDF; 400 — если файл повреждён или это скан."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать PDF")
    if len(text.strip()) < MIN_TEXT_CHARS:
        raise HTTPException(status_code=400, detail="PDF не содержит текст (возможно, это скан)")
    return text


def run_analysis(job_id: str, resume_text: str, input_type: str) -> dict:
    """Общий анализ для PDF и текста + сохранение результата в SQLite."""
    job = JOBS[job_id]
    result = analyze(
        resume_text=resume_text,
        job_skills=job.skills,
        job_description=job.text_en,    # английский текст: TF-IDF обучен на английских резюме
    )
    result["analysis_id"] = save_analysis(job_id, result, input_type, len(resume_text))
    return result

# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/jobs")
async def list_jobs():
    return [
        {
            "id": j.id,
            "title": j.title,
            "team": j.team,
            "location": j.location,
            "type": j.type,
            "level": j.level,
            "description": j.description,
        }
        for j in JOBS.values()
    ]

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    j = get_job_or_404(job_id)
    return {
        "id": j.id,
        "title": j.title,
        "team": j.team,
        "location": j.location,
        "type": j.type,
        "level": j.level,
        "description": j.description,
        "responsibilities": j.responsibilities,
        "requirements": j.requirements,
        "nice_to_have": j.nice_to_have,
    }

@app.get("/api/model-info")
async def model_info():
    """Какие модели и версии библиотек загружены (проверка пункта 1)."""
    return MODEL_INFO

@app.post("/api/analyze/{job_id}")
async def analyze_resume(job_id: str, file: UploadFile = File(...)):
    get_job_or_404(job_id)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    content = await file.read()
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="Файл больше 5 МБ")

    resume_text = pdf_to_text(content)
    return JSONResponse(run_analysis(job_id, resume_text, "pdf"))

@app.post("/api/analyze-text/{job_id}")
async def analyze_resume_text(job_id: str, body: ResumeText):
    """Прямой ввод текста резюме (пункт 14)."""
    get_job_or_404(job_id)
    text = body.text.strip()
    if len(text) < MIN_TEXT_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"Слишком короткий текст (минимум {MIN_TEXT_CHARS} символов)")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail="Слишком длинный текст")
    return JSONResponse(run_analysis(job_id, text, "text"))
