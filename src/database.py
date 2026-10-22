
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=True)

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("MONGO_DB_NAME", "research_rag")

if not MONGO_URI:
    raise ValueError("MONGO_URI is missing from .env")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

db = client[DATABASE_NAME]

documents_collection = db["documents"]
chunks_collection = db["chunks"]


def check_connection():
    client.admin.command("ping")
    return True
