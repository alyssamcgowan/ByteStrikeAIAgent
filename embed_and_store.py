from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks.")

embeddingModel = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", 
encode_kwargs={"normalize_embeddings": True})

vectorstore = Chroma(
    persist_directory=persistDir,
    embedding_function=embeddingModel
)

batch_size = 64 
for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)

print(f"Stored all embeddings persistently in: {persistDir}")
