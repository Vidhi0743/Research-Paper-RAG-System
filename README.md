
# Research Paper RAG System

A Retrieval-Augmented Generation (RAG) application built using Python, FastAPI, MongoDB Atlas Local, Sentence Transformers, and Ollama.

The system answers questions using a predefined collection of three research papers. It retrieves relevant passages through MongoDB Vector Search and generates grounded answers using a locally running language model.

## 1. Project Objective

Build a research-paper question-answering system that:

- Processes research papers in advance.
- Extracts and divides PDF content into smaller chunks.
- Generates vector embeddings for each chunk.
- Stores documents, chunks, and embeddings in MongoDB.
- Retrieves relevant chunks using semantic vector search.
- Generates answers using a local LLM.
- Displays answers and supporting sources through a web interface.

Users do not need to upload documents while asking questions.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Backend API | FastAPI |
| Database | MongoDB Atlas Local |
| Vector search | MongoDB Vector Search |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding dimensions | 384 |
| Language model | Llama 3.2 (3B) |
| LLM runtime | Ollama |
| Frontend | HTML, CSS, JavaScript |
| Container runtime | Docker Desktop |

## 3. System Architecture

### Offline Knowledge Base Preparation

Research Papers (PDF)
        |
        v
Text Extraction
        |
        v
Text Chunking
        |
        v
Embedding Generation
        |
        v
MongoDB Storage
(Documents + Chunks + Embeddings)

### Online Question Answering

User Question
        |
        v
FastAPI Backend
        |
        v
Question Embedding
        |
        v
MongoDB Vector Search
        |
        v
Top Relevant Chunks
        |
        v
Ollama (Llama 3.2)
        |
        v
Generated Answer + Supporting Sources
        |
        v
Web Interface

## 4. Knowledge Base

The current knowledge base contains:

- 3 research papers
- 136 text chunks
- 136 stored embeddings
- 384 dimensions per embedding

Knowledge base ID:

`research_papers_v1`

MongoDB database:

`research_rag`

Main collections:

- `documents`
- `chunks`

Vector search index:

`research_vector_index`

## 5. Project Structure

```text
RAG-Document-Ingestion/
|
|-- app/
|   |-- api.py
|   |-- templates/
|   |   |-- index.html
|   |-- static/
|       |-- style.css
|       |-- script.js
|
|-- src/
|   |-- database.py
|   |-- document_loader.py
|   |-- embedder.py
|   |-- retriever.py
|   |-- generator.py
|
|-- scripts/
|   |-- prepare_knowledge_base.py
|   |-- migrate_to_atlas_local.py
|   |-- create_vector_index.py
|   |-- test_vector_search.py
|
|-- data/
|   |-- research_papers/
|       |-- paper1.pdf
|       |-- paper2.pdf
|       |-- paper3.pdf
|
|-- .env
|-- requirements.txt
|-- README.md
```

## 6. Environment Configuration

The MongoDB Atlas Local connection URI is stored in `.env`.

Example:

```dotenv
ATLAS_LOCAL_URI=mongodb://127.0.0.1:<DOCKER_PORT>/?directConnection=true
```

Replace `<DOCKER_PORT>` with the port currently mapped to the Atlas Local container.

Do not commit `.env` files containing private credentials.

## 7. Starting the Application

### Step 1: Start Docker Desktop

Ensure Docker Desktop is running.

### Step 2: Start MongoDB Atlas Local

Open PowerShell:

```powershell
atlas local list
```

If the existing deployment is stopped:

```powershell
atlas local start local7555
```

Check its current port:

```powershell
docker ps -a --filter "name=local7555"
```

Update `ATLAS_LOCAL_URI` in `.env` if the mapped port has changed.

Do not create a new deployment each time.

### Step 3: Start Ollama

Ensure Ollama is running and the required model is available:

```powershell
ollama list
```

Required model:

```text
llama3.2:3b
```

If it is not installed:

```powershell
ollama pull llama3.2:3b
```

### Step 4: Activate the Python Virtual Environment

Navigate to the project folder and run:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Start FastAPI

```powershell
python -m uvicorn app.api:app --reload
```

### Step 6: Open the Application

Frontend:

http://127.0.0.1:8000/

Swagger API documentation:

http://127.0.0.1:8000/docs

Database status:

http://127.0.0.1:8000/database/status

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Displays the chatbot interface |
| GET | `/database/status` | Checks database and vector index status |
| POST | `/ask` | Retrieves passages and generates an answer |

Example request to `/ask`:

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "top_k": 3
}
```

The response includes the generated answer, model name, and supporting source passages.

## 9. Retrieval and Answer Generation

1. The user submits a research-related question.
2. The embedding model converts the question into a 384-dimensional vector.
3. MongoDB searches the stored embeddings using cosine similarity.
4. Relevant chunks are retrieved from the predefined knowledge base.
5. Retrieved text is supplied as context to Llama 3.2.
6. The generated answer and supporting sources are returned to the frontend.

## 10. Important Notes

- The research papers are processed in advance.
- Users do not upload or chunk documents during question answering.
- Embeddings are stored in MongoDB and reused.
- Only the incoming question requires a new embedding.
- The LLM runs locally through Ollama.
- The existing MongoDB deployment and stored knowledge base should be preserved.
- Knowledge base preparation and migration scripts are not required for normal application startup.

## 11. Current Project Status

Completed:

- PDF text extraction
- Document chunking
- Embedding generation
- MongoDB storage
- Atlas Local deployment
- Native vector search index
- Semantic retrieval
- Ollama integration
- FastAPI backend
- Chatbot frontend
- Source references
- Environment-based MongoDB configuration
