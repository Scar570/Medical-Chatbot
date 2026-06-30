"""
Retrieval Evaluation Script for Medical Chatbot RAG Pipeline
================================================================
Measures: Hit Rate@k  -  for each test question, did the correct
medical topic actually get retrieved from Pinecone in the top-k results?

This is a REAL, defensible metric you can quote to an interviewer:
"My retriever achieved X% Hit Rate@3 on a 25-question test set built
from the actual source PDF."

HOW TO RUN:
1. Place this file in your Medical-Chatbot repo root (same level as app.py)
2. Place eval_test_set.csv in the same folder
3. Make sure your .env has PINECONE_API_KEY set and the index is already
   populated (you've run store_index.py at least once)
4. Run: python eval_retrieval.py

OUTPUT:
- Per-question pass/fail printed to console
- Final Hit Rate@k score
- A results CSV you can screenshot/attach as evidence
"""

import os
import csv
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings  # or langchain_community if older version

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"  # change if your index name differs
TOP_K = 3  # how many chunks to retrieve per query

def load_test_set(path="eval_test_set.csv"):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": TOP_K})

    test_set = load_test_set("eval_test_set_10.csv")
    results = []
    hits = 0

    for row in test_set:
        question = row["question"]
        expected_topic = row["expected_topic"].lower()

        retrieved_docs = retriever.invoke(question)
        retrieved_text = " ".join(doc.page_content.lower() for doc in retrieved_docs)

        # Hit = the expected topic name shows up somewhere in the top-k retrieved chunks
        hit = expected_topic in retrieved_text
        hits += int(hit)

        status = "HIT " if hit else "MISS"
        print(f"[{status}] Q{row['id']}: {question}")

        results.append({
            "id": row["id"],
            "question": question,
            "expected_topic": row["expected_topic"],
            "hit": hit,
        })

    hit_rate = hits / len(test_set) * 100

    print("\n" + "=" * 50)
    print(f"Hit Rate@{TOP_K}: {hits}/{len(test_set)} = {hit_rate:.1f}%")
    print("=" * 50)

    with open("eval_retrieval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "expected_topic", "hit"])
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved detailed results to eval_retrieval_results.csv")

if __name__ == "__main__":
    main()
