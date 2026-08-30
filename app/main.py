from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import SupportAgent


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Aster & Row — AI Support")

agent = SupportAgent()


# Serve CSS, JavaScript and other static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(request: ChatRequest):
    result = agent.chat(request.message)

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "route": result.get("route"),
        "handoff": result.get("handoff", False),
        "tool_result": result.get("tool_result"),
    }


@app.post("/api/new-chat")
def new_chat():
    agent.memory.clear()
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"status": "ok"}