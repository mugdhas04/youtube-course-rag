import ollama
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="dsa_course")

# Test questions with the video title we EXPECT the answer to come from
# (based on topics covered in each video)
test_cases = [
    {"question": "what is time complexity", "expected_keyword": "Time Complexity"},
    {"question": "what is big o notation", "expected_keyword": "Big O"},
    {"question": "how do you insert an element in an array", "expected_keyword": "Insertion"},
    {"question": "what is a linked list", "expected_keyword": "Linked List"},
    {"question": "difference between linear and binary search", "expected_keyword": "Search"},
    {"question": "how to delete a node from a linked list", "expected_keyword": "Deletion"},
    {"question": "what is asymptotic notation", "expected_keyword": "Asymptotic"},
    {"question": "what is a circular linked list", "expected_keyword": "Circular"},
    {"question": "best case and worst case analysis", "expected_keyword": "Best Case"},
    {"question": "what is an abstract data type", "expected_keyword": "Abstract Data Type"},
]

def evaluate():
    hits = 0
    results_log = []

    for case in test_cases:
        response = ollama.embed(model="nomic-embed-text", input=case["question"])
        question_embedding = response["embeddings"][0]

        results = collection.query(query_embeddings=[question_embedding], n_results=3)
        retrieved_titles = [m["video_title"] for m in results["metadatas"][0]]

        # Check if the expected keyword appears in ANY of the top-3 retrieved video titles
        hit = any(case["expected_keyword"].lower() in title.lower() for title in retrieved_titles)
        if hit:
            hits += 1

        results_log.append({
            "question": case["question"],
            "expected": case["expected_keyword"],
            "retrieved": retrieved_titles,
            "hit": hit
        })

    accuracy = (hits / len(test_cases)) * 100

    print("=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    for r in results_log:
        status = "✓ HIT" if r["hit"] else "✗ MISS"
        print(f"\n[{status}] Q: {r['question']}")
        print(f"   Expected topic: {r['expected']}")
        print(f"   Top-3 retrieved from: {r['retrieved']}")

    print("\n" + "=" * 70)
    print(f"RETRIEVAL ACCURACY (Hit Rate @ Top-3): {hits}/{len(test_cases)} = {accuracy:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    evaluate()