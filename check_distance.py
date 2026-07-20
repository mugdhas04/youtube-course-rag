import ollama
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="playlist_PLfqMhTWNBTe3LtFWcvwpqTkUSlB32kJop")

question = "What is a variable in Java?"
response = ollama.embed(model="nomic-embed-text", input=question)
embedding = response["embeddings"][0]

results = collection.query(query_embeddings=[embedding], n_results=3)
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"Distance: {dist:.3f} | {meta['video_title']} at {meta['start']}")
    print(f"Text: {doc[:100]}...")
    print()