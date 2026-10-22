
import os

from dotenv import load_dotenv
from pymongo import MongoClient

from src.embedder import generate_embedding


# Load environment variables from the project's .env file.
load_dotenv()

ATLAS_LOCAL_URI = os.getenv("ATLAS_LOCAL_URI")

if not ATLAS_LOCAL_URI:
    raise RuntimeError(
        "ATLAS_LOCAL_URI is missing from the .env file."
    )


DATABASE_NAME = "research_rag"
COLLECTION_NAME = "chunks"

KNOWLEDGE_BASE_ID = "research_papers_v1"
VECTOR_INDEX_NAME = "research_vector_index"
EMBEDDING_DIMENSIONS = 384


client = MongoClient(
    ATLAS_LOCAL_URI,
    serverSelectionTimeoutMS=10000
)

chunks_collection = client[DATABASE_NAME][COLLECTION_NAME]


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 5
) -> list[dict]:
    """
    Retrieve relevant research-paper chunks using
    MongoDB's native Vector Search index.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Generate an embedding for the user's question.
    question_vector = generate_embedding(question.strip())

    if len(question_vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            "Question embedding has incorrect dimensions."
        )

    # Retrieve matching passages from MongoDB Atlas Local.
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": question_vector,
                "numCandidates": max(100, top_k * 20),
                "limit": top_k,
                "filter": {
                    "knowledge_base_id": KNOWLEDGE_BASE_ID
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "source": 1,
                "page_number": 1,
                "section": 1,
                "text": 1,
                "similarity_score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    results = list(chunks_collection.aggregate(pipeline))

    for chunk in results:
        chunk["similarity_score"] = round(
            float(chunk.get("similarity_score", 0)),
            4
        )

    return results
