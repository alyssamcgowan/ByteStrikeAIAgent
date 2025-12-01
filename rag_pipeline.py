# rag_pipeline.py
import os
from dotenv import load_dotenv
from typing import List
from huggingface_hub import InferenceClient

from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "chroma_store/"

# =========================================================
# Lightweight Document class
# =========================================================
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# =========================================================
# HuggingFace Inference API — Embeddings Only
# =========================================================
class HFInferenceEmbeddings:
    """Embedding class using HuggingFace Inference API (InferenceClient)."""

    def __init__(self, model_name: str):
        self.model_name = model_name

        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set!")

        # Uses router-based inference
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )

    def _embed(self, text: str):
        # HF API returns a nested list (batch_size x dim)
        result = self.client.feature_extraction(
            text,
            model=self.model_name
        )

        # If shape = [ [vector] ], flatten it
        if isinstance(result, list) and isinstance(result[0], list):
            return result[0]

        return result

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

# =========================================================
# Instantiate embeddings + Chroma store
# =========================================================
print("Using HF Inference API embeddings (router)...", flush=True)
embedding_model = HFInferenceEmbeddings(
    model_name=os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
)

print("Loading vector store...", flush=True)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)
print(f"Vector store loaded: {vectorstore._collection.count()} documents", flush=True)

# =========================================================
# Multi-stage retrieval (NO reranking)
# =========================================================
async def retrieve_docs(query: str, k_initial: int = 25, k_final: int = 7) -> List[Document]:
    """
    Dense retrieval only — no reranker.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_initial})
    docs = retriever.invoke(query)

    if not docs:
        return []

    # Just truncate the top-k results from Chroma
    top_docs = [
        Document(page_content=d.page_content, metadata=d.metadata)
        for d in docs[:k_final]
    ]

    return top_docs

# =========================================================
# Utility to get combined context
# =========================================================
async def get_context(query: str, k_initial: int = 40, k_final: int = 7) -> str:
    docs = await retrieve_docs(query, k_initial, k_final)
    return "\n\n".join([d.page_content for d in docs]) if docs else ""
