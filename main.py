import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pypdf

from jobs import JOBS
from nlp_pipeline import analyze

app = FastAPI(title="Resume Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Страницы ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_page(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    with open("static/job.html", encoding="utf-8") as f:
        return f.read()

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
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    j = JOBS[job_id]
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

@app.post("/api/analyze/{job_id}")
async def analyze_resume(job_id: str, file: UploadFile = File(...)):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Только PDF файлы")

    # Читаем PDF
    content = await file.read()
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        resume_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать PDF")

    if len(resume_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="PDF не содержит текст (возможно, это скан)")

    job = JOBS[job_id]
    result = analyze(
        resume_text=resume_text,
        job_skills=job.skills,
        job_description=" ".join(job.requirements + job.responsibilities),
    )
    return JSONResponse(result)
