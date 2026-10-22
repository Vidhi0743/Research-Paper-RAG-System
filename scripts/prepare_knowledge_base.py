
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from pymongo import UpdateOne

# Allow running this file directly from the project root.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.database import (
    check_connection,
    documents_collection,
    chunks_collection,
)
from src.document_loader import load_document
from src.text_cleaner import clean_document
from src.research_chunker import chunk_research_document
from src.embedder import generate_embedding, MODEL_NAME


PAPERS_DIR = ROOT_DIR / "data" / "research_papers"
KNOWLEDGE_BASE_ID = "research_papers_v1"
EXPECTED_PDF_COUNT = 3
EMBEDDING_DIMENSIONS = 384


def file_hash(path):
    """Calculate a stable hash for a PDF."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def process_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path.name}")
    print("-" * 55)

    pdf_hash = file_hash(pdf_path)
    document_id = f"{KNOWLEDGE_BASE_ID}:{pdf_path.name}"

    # Extract and clean PDF text.
    document = load_document(pdf_path)
    cleaned_document = clean_document(document)

    if not cleaned_document["pages"]:
        raise ValueError("No extractable text found in this PDF.")

    # Generate chunks using the existing research-paper chunker.
    chunks = chunk_research_document(cleaned_document)

    if not chunks:
        raise ValueError("No chunks were generated.")

    print(f"Pages with text: {len(cleaned_document['pages'])}")
    print(f"Chunks generated: {len(chunks)}")

    existing = {
        record["chunk_id"]: record
        for record in chunks_collection.find(
            {
                "knowledge_base_id": KNOWLEDGE_BASE_ID,
                "document_id": document_id,
            },
            {
                "chunk_id": 1,
                "text": 1,
                "embedding": 1,
                "embedding_model": 1,
                "embedding_dimensions": 1,
            },
        )
    }

    operations = []
    generated_count = 0
    reused_count = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        old = existing.get(chunk_id)

        # Reuse valid existing embeddings on subsequent runs.
        reusable = (
            old is not None
            and old.get("text") == chunk["text"]
            and old.get("embedding_model") == MODEL_NAME
            and old.get("embedding_dimensions") == EMBEDDING_DIMENSIONS
            and isinstance(old.get("embedding"), list)
            and len(old["embedding"]) == EMBEDDING_DIMENSIONS
        )

        if reusable:
            embedding = old["embedding"]
            reused_count += 1
        else:
            embedding = generate_embedding(chunk["text"])
            generated_count += 1

        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Invalid embedding size for chunk {chunk_id}"
            )

        record = {
            **chunk,
            "knowledge_base_id": KNOWLEDGE_BASE_ID,
            "document_id": document_id,
            "document_hash": pdf_hash,
            "embedding": embedding,
            "embedding_model": MODEL_NAME,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "updated_at": datetime.now(timezone.utc),
        }

        operations.append(
            UpdateOne(
                {
                    "knowledge_base_id": KNOWLEDGE_BASE_ID,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                },
                {"$set": record},
                upsert=True,
            )
        )

    # Write after all embeddings for this paper are ready.
    chunks_collection.bulk_write(operations, ordered=True)

    # Remove outdated chunks belonging to this paper and knowledge base.
    current_ids = [chunk["chunk_id"] for chunk in chunks]

    chunks_collection.delete_many(
        {
            "knowledge_base_id": KNOWLEDGE_BASE_ID,
            "document_id": document_id,
            "chunk_id": {"$nin": current_ids},
        }
    )

    documents_collection.update_one(
        {
            "knowledge_base_id": KNOWLEDGE_BASE_ID,
            "document_id": document_id,
        },
        {
            "$set": {
                "knowledge_base_id": KNOWLEDGE_BASE_ID,
                "document_id": document_id,
                "filename": pdf_path.name,
                "source_path": str(pdf_path),
                "file_hash": pdf_hash,
                "page_count": len(cleaned_document["pages"]),
                "chunk_count": len(chunks),
                "embedding_model": MODEL_NAME,
                "status": "completed",
                "updated_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )

    print(f"New embeddings generated: {generated_count}")
    print(f"Existing embeddings reused: {reused_count}")
    print(f"Chunks stored: {len(chunks)}")
    print("[SUCCESS] Paper completed")

    return len(chunks)


def main():
    print("\nRESEARCH PAPER RAG - KNOWLEDGE BASE PREPARATION")
    print("=" * 55)

    if not PAPERS_DIR.is_dir():
        raise FileNotFoundError(
            f"Research papers folder not found: {PAPERS_DIR}"
        )

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    print(f"PDFs found: {len(pdf_files)}")

    if len(pdf_files) != EXPECTED_PDF_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PDF_COUNT} PDFs, "
            f"but found {len(pdf_files)}. "
            "No database changes were made."
        )

    check_connection()
    print("[SUCCESS] MongoDB connected")

    total_chunks = 0
    completed = 0

    for pdf_path in pdf_files:
        try:
            total_chunks += process_pdf(pdf_path)
            completed += 1
        except Exception as error:
            print(f"[ERROR] {pdf_path.name}: {error}")
            raise

    print("\n" + "=" * 55)
    print("PREPROCESSING SUMMARY")
    print("=" * 55)
    print(f"PDFs processed: {completed}")
    print(f"Chunks generated: {total_chunks}")

    stored_count = chunks_collection.count_documents(
        {"knowledge_base_id": KNOWLEDGE_BASE_ID}
    )

    print(f"Knowledge-base chunks in MongoDB: {stored_count}")
    print("\n[SUCCESS] Preprocessing completed!")


if __name__ == "__main__":
    main()
