
from pymongo import MongoClient, ReplaceOne

KB_ID = "research_papers_v1"

# Existing MongoDB Community Server (source)
SOURCE_URI = "mongodb://127.0.0.1:27017/"

# New Atlas Local deployment (destination)
TARGET_URI = "mongodb://127.0.0.1:42451/?directConnection=true"

source_client = MongoClient(SOURCE_URI, serverSelectionTimeoutMS=10000)
target_client = MongoClient(TARGET_URI, serverSelectionTimeoutMS=10000)

try:
    source_client.admin.command("ping")
    target_client.admin.command("ping")

    source_db = source_client["research_rag"]
    target_db = target_client["research_rag"]

    source_documents = list(
        source_db.documents.find({"knowledge_base_id": KB_ID})
    )

    source_chunks = list(
        source_db.chunks.find({"knowledge_base_id": KB_ID})
    )

    print("Source documents:", len(source_documents))
    print("Source chunks:", len(source_chunks))

    # Safety checks: stop if the source data is unexpected.
    assert len(source_documents) == 3, "Expected exactly 3 documents"
    assert len(source_chunks) == 136, "Expected exactly 136 chunks"

    for chunk in source_chunks:
        embedding = chunk.get("embedding")
        assert isinstance(embedding, list), "Missing embedding"
        assert len(embedding) == 384, "Incorrect embedding dimensions"

    # Preserve original MongoDB _id values and embeddings.
    for collection_name, records in [
        ("documents", source_documents),
        ("chunks", source_chunks),
    ]:
        operations = [
            ReplaceOne(
                {"_id": record["_id"]},
                record,
                upsert=True
            )
            for record in records
        ]

        if operations:
            target_db[collection_name].bulk_write(
                operations,
                ordered=False
            )

    # Verify copied records and their embeddings.
    target_documents = target_db.documents.count_documents(
        {"knowledge_base_id": KB_ID}
    )

    target_chunks = list(
        target_db.chunks.find({"knowledge_base_id": KB_ID})
    )

    valid_embeddings = sum(
        isinstance(chunk.get("embedding"), list)
        and len(chunk["embedding"]) == 384
        for chunk in target_chunks
    )

    print("\n--- Atlas Local verification ---")
    print("Documents:", target_documents)
    print("Chunks:", len(target_chunks))
    print("Valid 384-dimensional embeddings:", valid_embeddings)

    assert target_documents == 3
    assert len(target_chunks) == 136
    assert valid_embeddings == 136

    print("\n[SUCCESS] Migration verified!")

finally:
    source_client.close()
    target_client.close()
