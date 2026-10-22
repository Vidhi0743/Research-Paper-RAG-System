
from pymongo import MongoClient
from src.embedder import generate_embedding

client = MongoClient(
    "mongodb://127.0.0.1:42451/?directConnection=true",
    serverSelectionTimeoutMS=10000
)

collection = client["research_rag"]["chunks"]

question = (
    "How does retrieval-augmented generation combine "
    "parametric and non-parametric memory?"
)

# Embed only the new question.
# The 136 stored document embeddings are reused.
query_embedding = generate_embedding(question)

pipeline = [
    {
        "$vectorSearch": {
            "index": "research_vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 5,
            "filter": {
                "knowledge_base_id": "research_papers_v1"
            }
        }
    },
    {
        "$project": {
            "_id": 0,
            "source": 1,
            "page_number": 1,
            "section": 1,
            "text": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

try:
    indexes = list(collection.list_search_indexes(
        name="research_vector_index"
    ))

    if not indexes or not indexes[0].get("queryable"):
        print("Vector index is not queryable yet.")
        print("Current index details:", indexes)
        raise SystemExit(1)

    results = list(collection.aggregate(pipeline))

    print("\nQUESTION:", question)
    print("RESULTS FOUND:", len(results))

    for number, result in enumerate(results, start=1):
        print(f"\n--- RESULT {number} ---")
        print("Source:", result.get("source"))
        print("Page:", result.get("page_number"))
        print("Section:", result.get("section"))
        print("Score:", round(result.get("score", 0), 4))
        print("Text:", result.get("text", "")[:350])

finally:
    client.close()
