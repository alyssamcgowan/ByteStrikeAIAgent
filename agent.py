import langchain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
persistDir = "chroma_store/"

embeddingModel = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

def multistage_retrieve(query, vectorstore, k_initial=40, k_final=7):
    """
    Stage 1: Retrieve k_initial dense-vector candidates.
    Stage 2: Re-rank them with a cross-encoder.
    Stage 3: Return top k_final documents.
    """
    # Stage 1 — high-recall retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_initial})
    initial_docs = retriever.invoke(query)

    if not initial_docs:
        return []

    # Prepare pairs for cross-encoder
    pairs = [(query, doc.page_content) for doc in initial_docs]

    # Stage 2 — relevance scoring
    scores = reranker.predict(pairs)

    # Sort candidate docs by score
    ranked = sorted(
        zip(scores, initial_docs),
        key=lambda x: x[0],
        reverse=True
    )

    # Stage 3 — top-k
    final_docs = [doc for score, doc in ranked[:k_final]]
    return final_docs


vectorstore = Chroma(
    persist_directory=persistDir,
    embedding_function=embeddingModel
)

print("Number of stored vectors:", vectorstore._collection.count())

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

template = """
You are ByteStrike’s internal AI assistant, supporting the founder in growing and managing the company. 
Queries referring to "us," "we," "our," etc. refer to ByteStrike and its team. 

Your responsibilities include:
Drafting professional communications (emails, investor updates, outreach messages, etc.) in the founder’s tone and writing style, based on the examples in the internal document store.
Creating job descriptions, team workstreams, and project outlines.
Researching and summarizing information about potential hires (e.g., CTO candidates), investors (VCs), competitors, and relevant market or technology trends.
Helping organize company operations, strategy, and internal documentation efficiently.

Instructions:
Use the context from ByteStrike’s internal document vector store to inform answers and emulate the founder’s style.
You may reason, infer, or draw upon general knowledge beyond the retrieved context to provide helpful guidance or suggestions.
Do not fabricate information about the company.
Avoid generic statements and generalizations -- be specific to ByteStrike when possible.

Be concise, professional, and accurate while reflecting the founder’s voice.
Context:
{context}
Question:
{question}
"""

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    api_key=api_key
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

query = "What is our business plan?"


retrieved_docs = multistage_retrieve(query, vectorstore)
context = "\n\n".join([d.page_content for d in retrieved_docs])
final_prompt = prompt.format(context=context, question=query)

result = llm.invoke(final_prompt)
print(result.content)
