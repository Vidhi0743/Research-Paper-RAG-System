
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from src.generator import generate_rag_answer
from src.retriever import chunks_collection, KNOWLEDGE_BASE_ID


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Research Paper RAG System",
    description=(
        "Question-answering system using three research papers, "
        "MongoDB Vector Search and Ollama Llama 3.2."
    ),
    version="2.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# --------------------------------------------------
# Request validation
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about the research papers"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of research-paper chunks to retrieve"
    )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():

    index_path = TEMPLATE_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend template not found."
        )

    return HTMLResponse(
        content=index_path.read_text(encoding="utf-8")
    )


# --------------------------------------------------
# Database status
# --------------------------------------------------

@app.get("/database/status")
async def database_status():

    def check_status():

        chunks_collection.database.client.admin.command("ping")

        total_chunks = chunks_collection.count_documents({
            "knowledge_base_id": KNOWLEDGE_BASE_ID
        })

        indexes = list(
            chunks_collection.list_search_indexes(
                name="research_vector_index"
            )
        )

        vector_index_ready = bool(
            indexes and indexes[0].get("queryable")
        )

        return {
            "connected": True,
            "database": "research_rag",
            "knowledge_base": KNOWLEDGE_BASE_ID,
            "total_chunks": total_chunks,
            "vector_index_ready": vector_index_ready
        }

    try:
        return await run_in_threadpool(check_status)

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="MongoDB Atlas Local is unavailable."
        )


# --------------------------------------------------
# RAG question-answering endpoint
# --------------------------------------------------

@app.post("/ask")
async def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty."
        )

    try:
        result = await run_in_threadpool(
            generate_rag_answer,
            question,
            request.top_k
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="MongoDB Vector Search is unavailable."
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )
