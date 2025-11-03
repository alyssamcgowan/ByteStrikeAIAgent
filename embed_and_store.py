from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from tqdm import tqdm


dataDir = "Byte Strike/"
persistDir = "chroma_store/"

docs = []
for filename in os.listdir(dataDir):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(dataDir, filename)
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
print(f"Loaded {len(docs)} documents from {dataDir}")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

# embeddings = OpenAIEmbeddings()
embeddingModel = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", 
encode_kwargs={"normalize_embeddings": True})

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddingModel,
    persist_directory=persistDir
)

batch_size = 64 
for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)
    vectorstore.persist()  

print(f"Stored all embeddings persistently in: {persistDir}")
