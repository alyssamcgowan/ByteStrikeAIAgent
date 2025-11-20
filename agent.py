import langchain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
persistDir = "chroma_store/"

embeddingModel = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
) #base -> small b/c bad ram

vectorstore = Chroma(
    persist_directory=persistDir,
    embedding_function=embeddingModel
)
first_vector = vectorstore._collection.get(include=["embeddings"], ids=[vectorstore._collection.get()["ids"][0]])["embeddings"][0]
print("Embedding dimension of first vector:", len(first_vector))


print("Number of stored vectors:", vectorstore._collection.count())

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

template = """
You are ByteStrike’s internal AI assistant, supporting the founder in growing and managing the company.

Your responsibilities include:
Drafting professional communications (emails, investor updates, outreach messages, etc.) in the founder’s tone and writing style, based on the examples in the internal document store.
Creating job descriptions, team workstreams, and project outlines.
Researching and summarizing information about potential hires (e.g., CTO candidates), investors (VCs), competitors, and relevant market or technology trends.
Helping organize company operations, strategy, and internal documentation efficiently.

Instructions:
Use the context from ByteStrike’s internal document vector store to inform answers and emulate the founder’s style.
You may reason, infer, or draw upon general knowledge beyond the retrieved context to provide helpful guidance or suggestions.
Do not fabricate information about the company. If a question concerns ByteStrike-specific facts not contained in the documents, respond with:
“The available documents do not contain that information.”
Avoid generic statements and generalizations -- be specific to ByteStrike when possible.

Be concise, professional, and accurate while reflecting the founder’s voice.
Context:
{context}
Question:
{question}
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff", 
    chain_type_kwargs={"prompt": prompt}
)

# query = "Who is the CEO of ByteStrike?"

#print top k documents retrieved to be used for this query.

# retrieved_docs = retriever.invoke(query)
# print("\n--- Retrieved Contexts ---")
# if not retrieved_docs:
#     print("No documents were retrieved for this query.")
# else:
#     for i, doc in enumerate(retrieved_docs, start=1):
#         print(f"\n[Document {i}]")
#         print(doc.page_content[:500])  # print first 500 chars
#         print("---")
# query = input("What do you want to search: ")

# result = qa_chain.invoke({"query": query})
# print(result["result"])


def agent(query: str) -> str:
    """Run a query against the ByteStrike RAG system and return the result."""
    result = qa_chain.invoke({"query": query})
    return result["result"]

if __name__ == "__main__":
    # simple manual test
    test_query = input("What do you want to search? ")
    print(agent(test_query))

