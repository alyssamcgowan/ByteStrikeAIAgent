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

embeddingModel = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

vectorstore = Chroma(
    persist_directory=persistDir,
    embedding_function=embeddingModel
)
print("Number of stored vectors:", vectorstore._collection.count())

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

template = """You are an internal AI assistant that supports the startup ByteStrike by helping draft emails,
answer operational questions, and perform general business tasks efficiently.
Learn the founder's voice and style of writing for emails and other documents.
You have access only to the following context from the company's internal document vector store.
Use only the provided context from the company's information store to answer the user's question. 
If you don't know, say you don't know."
Be concise and accurate.
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

query = "Who is the founder?"

# retrieved_docs = retriever.invoke(query)
# print("\n--- Retrieved Contexts ---")
# if not retrieved_docs:
#     print("No documents were retrieved for this query.")
# else:
#     for i, doc in enumerate(retrieved_docs, start=1):
#         print(f"\n[Document {i}]")
#         print(doc.page_content[:500])  # print first 500 chars
#         print("---")

result = qa_chain.invoke({"query": query})
print(result["result"])
