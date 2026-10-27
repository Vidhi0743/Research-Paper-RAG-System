
from pathlib import Path
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. MODEL CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"


# --------------------------------------------------
# 2. LOAD MODEL ONCE (PERSISTENT LOCAL MODEL)
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model from local storage.

    The model is initialized only once per Python process.
    Subsequent calls reuse the same model object.

    No internet download is performed.
    """

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Local embedding model not found at: {MODEL_PATH}\n"
            "Download and save the model before starting FastAPI."
        )

    print("[EMBEDDER] Loading embedding model from local storage...")

    model = SentenceTransformer(
        str(MODEL_PATH),
        local_files_only=True
    )

    print("[EMBEDDER] Model loaded successfully.")

    return model


# Initialize model once
model = get_embedding_model()

# Access the model tokenizer
tokenizer = model.tokenizer

# Calculate maximum usable tokens
MAX_TOKENS = (
    model.max_seq_length
    - tokenizer.num_special_tokens_to_add(pair=False)
)


# --------------------------------------------------
# 3. SPLIT LONG TEXT INTO TOKEN WINDOWS
# --------------------------------------------------

def split_into_token_windows(text: str) -> list[str]:
    """
    Split long text into windows that fit the model's
    maximum token limit.

    No content is intentionally truncated.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    windows = []

    for start in range(0, len(token_ids), MAX_TOKENS):

        window_ids = token_ids[
            start:start + MAX_TOKENS
        ]

        window_text = tokenizer.decode(
            window_ids,
            skip_special_tokens=True
        )

        windows.append(window_text)

    return windows


# --------------------------------------------------
# 4. GENERATE SINGLE EMBEDDING
# --------------------------------------------------

def generate_embedding(text: str) -> list[float]:
    """
    Generate one normalized 384-dimensional embedding.

    For long text:
    1. Split into token windows.
    2. Generate embeddings for every window.
    3. Calculate mean embedding.
    4. Normalize the final vector.
    """

    windows = split_into_token_windows(text)

    # Actual model inference happens here
    window_embeddings = model.encode(
        windows,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Combine embeddings from all windows
    combined = np.mean(
        window_embeddings,
        axis=0
    )

    # Calculate vector magnitude
    norm = np.linalg.norm(combined)

    if norm == 0:
        raise ValueError(
            "Cannot normalize a zero embedding."
        )

    # Normalize final embedding
    combined = combined / norm

    return combined.tolist()


# --------------------------------------------------
# 5. GENERATE MULTIPLE EMBEDDINGS
# --------------------------------------------------

def generate_embeddings(
    texts: list[str]
) -> list[list[float]]:
    """
    Generate one embedding per input text.

    Uses the already loaded embedding model.
    """

    return [
        generate_embedding(text)
        for text in texts
    ]
