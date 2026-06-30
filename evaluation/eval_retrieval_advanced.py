"""
Advanced Retrieval Evaluation: Precision@k, Recall@k, MRR
=============================================================
Fixes the false-positive problem found earlier: instead of matching just
the topic NAME (which can falsely match junk header/page-stamp chunks),
this checks for a distinctive PHRASE from the real definition text -
something a near-empty header chunk could never accidentally contain.

Run: python eval_retrieval_advanced.py
(same repo folder as your other eval scripts, .env with PINECONE_API_KEY)
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"
TOP_K = 3

# question -> a short, DISTINCTIVE phrase pulled from the real definition text
# (lowercased, partial match). Chosen to be specific enough that a junk/header
# chunk could not accidentally contain it.
TEST_SET = [
    ("Can you explain Abscess incision & drainage?", "infected skin nodule that contains pus"),
    ("What is Acromegaly and gigantism?", "abnormal release of a particular chemical from the pituitary gland"),
    ("What does the medical term Allergic purpura mean?", "allergic reaction of unknown origin causing red patches"),
    ("Can you explain Alpha1-adrenergic blockers?", "blocking the alpha1-receptors of vascular smooth muscle"),
    ("Can you explain Aminoglycosides?", "group of antibiotics that are used to treat certain bacterial infections"),
    ("Can you explain Antihelminthic drugs?", "used to treat parasitic infestations"),
    ("Can you explain Antiseptics?", "inhibits the growth and development of microorganisms"),
    ("Can you explain Appendicitis?", "inflammation of the appendix"),
    ("Can you explain Athletic heart syndrome?", "adaptation of an athlete"),
    ("Can you explain Bladder training?", "behavioral modification treatment technique for urinary incontinence"),
]

def normalize(text: str) -> str:
    """Collapse PDF line-wrap artifacts: de-hyphenate line breaks, then
    turn remaining newlines into spaces, then collapse repeated whitespace.
    Without this, phrases that happen to wrap across a line in the source
    PDF will never match even though the content is genuinely there."""
    text = text.replace("-\n", "")   # de-hyphenate words split across a line
    text = text.replace("\n", " ")   # remaining line breaks become spaces
    text = " ".join(text.split())    # collapse multiple spaces/tabs
    return text.lower()

def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": TOP_K})

    precisions = []
    recalls = []
    reciprocal_ranks = []

    for question, phrase in TEST_SET:
        docs = retriever.invoke(question)
        relevance_flags = [phrase.lower() in normalize(doc.page_content) for doc in docs]

        relevant_count = sum(relevance_flags)
        precision_at_k = relevant_count / TOP_K
        recall_at_k = 1.0 if relevant_count > 0 else 0.0  # single relevant chunk assumed per query

        rank = next((i + 1 for i, flag in enumerate(relevance_flags) if flag), None)
        rr = 1.0 / rank if rank else 0.0

        precisions.append(precision_at_k)
        recalls.append(recall_at_k)
        reciprocal_ranks.append(rr)

        status = f"rank {rank}" if rank else "NOT FOUND"
        print(f"[{status:10}] P@{TOP_K}={precision_at_k:.2f}  {question}")

    print("\n" + "=" * 60)
    print(f"Mean Precision@{TOP_K}: {sum(precisions)/len(precisions):.3f}")
    print(f"Mean Recall@{TOP_K}:    {sum(recalls)/len(recalls):.3f}")
    print(f"MRR:               {sum(reciprocal_ranks)/len(reciprocal_ranks):.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()