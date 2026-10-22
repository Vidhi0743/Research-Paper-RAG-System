
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer

MAX_TOKENS = model.max_seq_length - tokenizer.num_special_tokens_to_add(
    pair=False
)


def split_into_token_windows(text: str) -> list[str]:
    """
    Split long text into windows that fit the model's token limit.
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
        window_ids = token_ids[start:start + MAX_TOKENS]

        window_text = tokenizer.decode(
            window_ids,
            skip_special_tokens=True
        )

        windows.append(window_text)

    return windows


def generate_embedding(text: str) -> list[float]:
    """
    Generate one normalized 384-dimensional embedding.
    Long text is represented using all its token windows.
    """
    windows = split_into_token_windows(text)

    window_embeddings = model.encode(
        windows,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    combined = np.mean(window_embeddings, axis=0)

    norm = np.linalg.norm(combined)

    if norm == 0:
        raise ValueError("Cannot normalize a zero embedding.")

    combined = combined / norm

    return combined.tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate one embedding per input text.
    """
    return [
        generate_embedding(text)
        for text in texts
    ]
