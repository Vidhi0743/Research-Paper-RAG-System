
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)

mongo_uri = os.getenv("MONGO_URI")
database_name = os.getenv("MONGO_DB_NAME", "research_rag")

if not mongo_uri:
    raise ValueError("MONGO_URI is missing from .env")

client = None

try:
    print("Connecting to MongoDB...")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000
    )

    client.admin.command("ping")

    print("MongoDB connected successfully!")
    print("Database selected:", database_name)

except PyMongoError as error:
    print("MongoDB connection failed.")
    print("Error type:", type(error).__name__)
    print("Error details:", error)

finally:
    if client is not None:
        client.close()
