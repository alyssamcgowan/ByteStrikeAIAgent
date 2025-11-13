import os
import json
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


#config
persistDir = "chroma_store/"

driveDataDir = "ByteStrikeDrive/"
website_urls = ["https://byte-strike.com/"]
emailsJson = "emailsCleaned/cleaned_emails.json"


#load bytestrike website
web_loader = WebBaseLoader(website_urls)
web_docs = web_loader.load()
for d in web_docs:
    d.metadata["source"] = "website"
print(f"Loaded {len(web_docs)} pages from website")


#load google drive files
docs = []
for filename in os.listdir(driveDataDir):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(driveDataDir, filename)
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
print(f"Loaded {len(docs)} documents from {driveDataDir}")

#load cleaned emails json
emails = []
if os.path.exists(emailsJson):
    print(f"Loading JSON file: {emailsJson}")
    with open(emailsJson, "r", encoding="utf-8") as f:
        data = json.load(f)

    def flatten_json(obj, prefix=""):
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                lines.extend(flatten_json(v, f"{prefix}{k}."))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                lines.extend(flatten_json(v, f"{prefix}{i}."))
        else:
            lines.append(f"{prefix[:-1]}: {obj}")
        return lines

    flat_text = "\n".join(flatten_json(data))
    json_doc = Document(page_content=flat_text, metadata={"source": emailsJson})
    emails = [json_doc]
    print(f"Loaded JSON file with {len(flat_text.splitlines())} lines of data")
else:
    print(f"JSON file not found: {emailsJson}")


#combine all doc types
all_docs = web_docs + docs + emails
print(f"Total combined documents: {len(all_docs)}")

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
