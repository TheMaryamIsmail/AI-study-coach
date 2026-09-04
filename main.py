import io
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import run

app = FastAPI(title="AI Study Coach")

templates = Jinja2Templates(directory="templates")


class StudentRequest(BaseModel):
    name: str
    subject: str
    weak_topics: str
    days_remaining: int
    hours_per_day: float
    skill_level: str
    technique: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """GET endpoint to serve frontend HTML."""
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/generate-plan")
@app.post("/generate_plan")
async def generate_plan(data: StudentRequest):
    """POST endpoint to connect frontend input with run.py Gemini methods."""
    student_info = data.dict()
    technique = data.technique

    if technique == "zero_shot":
        result = run.run_zero_shot(student_info)
    elif technique == "few_shot":
        result = run.run_few_shot(student_info)
    elif technique == "chain_of_thought":
        result = run.run_chain_of_thought(student_info)
    else:
        result = "Invalid technique selected."

    return {"status": "success", "technique": technique, "plan": result}