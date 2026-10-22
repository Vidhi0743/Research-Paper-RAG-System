
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

client = MongoClient(
    "mongodb://127.0.0.1:42451/?directConnection=true",
    serverSelectionTimeoutMS=10000
)

collection = client["research_rag"]["chunks"]

index = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "cosine"
            },
            {
                "type": "filter",
                "path": "knowledge_base_id"
            }
        ]
    },
    name="research_vector_index",
    type="vectorSearch"
)

try:
    client.admin.command("ping")

    existing = [
        item["name"]
        for item in collection.list_search_indexes()
    ]

    if "research_vector_index" in existing:
        print("Vector index already exists.")
    else:
        result = collection.create_search_index(model=index)
        print("Created index:", result)

    print("\nCurrent Search indexes:")
    for item in collection.list_search_indexes():
        print(
            "Name:", item.get("name"),
            "| Type:", item.get("type"),
            "| Status:", item.get("status"),
            "| Queryable:", item.get("queryable")
        )

finally:
    client.close()

