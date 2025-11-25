# rag_pipeline.py
import os
from dotenv import load_dotenv
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

PERSIST_DIR = "chroma_store/"

# -----------------------------
# Lightweight Document class
# -----------------------------
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# -----------------------------
# Load embedding model & vector store
# -----------------------------
print("Loading embeddings model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

print("Loading vector store...")
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)
print(f"Vector store loaded: {vectorstore._collection.count()} documents")

# -----------------------------
# Load reranker (CrossEncoder)
# -----------------------------
print("Loading cross-encoder reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

# -----------------------------
# Multi-stage retrieval function
# -----------------------------
async def retrieve_docs(query: str, k_initial: int = 25, k_final: int = 7) -> List[Document]:
    """
    Multi-stage retrieval:
    1. Dense retrieval of k_initial candidates.
    2. Re-rank candidates using a CrossEncoder.
    3. Return top k_final documents as lightweight Document objects.
    """
    # Stage 1 — high-recall dense retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_initial})
    initial_docs = retriever.invoke(query)

    if not initial_docs:
        return []

    # Stage 2 — CrossEncoder re-ranking
    pairs = [(query, doc.page_content) for doc in initial_docs]
    scores = reranker.predict(pairs)

    # Rank documents by score
    ranked = sorted(
        zip(scores, initial_docs),
        key=lambda x: x[0],
        reverse=True
    )

    # Stage 3 — top-k
    top_docs = [Document(page_content=d.page_content, metadata=d.metadata) for score, d in ranked[:k_final]]

    return top_docs

# -----------------------------
# Optional: utility function to get combined context
# -----------------------------
async def get_context(query: str, k_initial: int = 40, k_final: int = 7) -> str:
    docs = await retrieve_docs(query, k_initial=k_initial, k_final=k_final)
    if not docs:
        return ""
    return "\n\n".join([d.page_content for d in docs])
