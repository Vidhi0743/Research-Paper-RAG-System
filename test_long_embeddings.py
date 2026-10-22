
from src.embedder import (
    split_into_token_windows,
    generate_embedding,
    MAX_TOKENS
)


short_text = "BERT is a bidirectional transformer model."

long_text = (
    "Retrieval augmented generation combines information retrieval "
    "with language generation to improve factual responses. "
) * 100


print("Maximum content tokens per window:", MAX_TOKENS)

print("\nSHORT TEXT TEST")
short_windows = split_into_token_windows(short_text)
short_vector = generate_embedding(short_text)

print("Token windows:", len(short_windows))
print("Embedding dimensions:", len(short_vector))


print("\nLONG TEXT TEST")
long_windows = split_into_token_windows(long_text)
long_vector = generate_embedding(long_text)

print("Token windows:", len(long_windows))
print("Embedding dimensions:", len(long_vector))

assert len(short_vector) == 384
assert len(long_vector) == 384
assert len(long_windows) > 1

print("\nAll embedding tests passed!")
