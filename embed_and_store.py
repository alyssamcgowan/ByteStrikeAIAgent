from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from tqdm import tqdm

dataDir = "Byte Strike/"
persistDir = "chroma_store/"

# ADD THIS FUNCTION
def find_pdf_files(directory):
    """Recursively find all PDF files in directory and subdirectories"""
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)
    return pdf_files

def update_vector_store():
    """Update existing vector store with new documents"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5", 
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=persistDir,
        embedding_function=embedding_model
    )
    
    # Find all PDF files
    pdf_files = find_pdf_files(dataDir)  # This now works
    
    if not pdf_files:
        print("No PDF files found to update.")
        return vectorstore
    
    print("Checking for new documents...")
    docs = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            file_docs = loader.load()
            docs.extend(file_docs)
            print(f"Loaded: {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    if not docs:
        print("No new documents to add.")
        return vectorstore
    
    # Split and add new documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    print(f"Adding {len(chunks)} new chunks to vector store...")
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    
    print("Vector store updated!")
    return vectorstore

def setup_vector_store(force_update=False):
    """Set up or load existing vector store, with automatic update detection"""
    if os.path.exists(persistDir) and os.listdir(persistDir) and not force_update:
        print("Loading existing vector store...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5", 
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore = Chroma(
            persist_directory=persistDir,
            embedding_function=embedding_model
        )
        
        # AUTO-DETECT NEW FILES - Add this check
        current_pdf_files = find_pdf_files(dataDir)
        
        if current_pdf_files:
            print(f"Found {len(current_pdf_files)} PDF files. Checking if update is needed...")
            # For simplicity, we'll always update if PDF files exist
            # In a real system, you'd compare with previously processed files
            print("Updating vector store with current files...")
            return update_vector_store()
        else:
            print("No PDF files found. Using existing vector store.")
            return vectorstore
    else:
        print("No existing vector store found or force update requested. Creating new one...")
        return create_vector_store()

def create_vector_store():
    """Create a new vector store from documents"""
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

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5", 
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persistDir
    )

    batch_size = 64 
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        vectorstore.persist()  

    print(f"Stored all embeddings persistently in: {persistDir}")
    return vectorstore

# This allows the file to both be imported AND run directly
if __name__ == "__main__":
    setup_vector_store()