
from src.embedder import generate_embedding, generate_embeddings


print("Testing single-text embedding...")

text = "BERT is a bidirectional transformer model."

vector = generate_embedding(text)

print("Input text:", text)
print("Embedding dimensions:", len(vector))
print("First 10 values:", vector[:10])


print("\nTesting multiple embeddings...")

sentences = [
    "BERT understands contextual relationships in language.",
    "Transformers use attention mechanisms.",
    "MongoDB stores documents in collections."
]

vectors = generate_embeddings(sentences)

print("Total embeddings:", len(vectors))

for index, embedding in enumerate(vectors, start=1):
    print(
        f"Sentence {index}: {len(embedding)} dimensions"
    )

print("\nEmbedding test completed successfully!")

