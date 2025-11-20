import os
import json
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import hashlib

def content_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

#config
persistDir = "fake/"
dataDir = "dataToEmbed/"

driveDataDir = os.path.join(dataDir, "ByteStrikeDrive/")
website_urls = ["https://byte-strike.com/"]
emailsJson = os.path.join(dataDir, "emailsCleaned/cleaned_emails.json")
slackDir = os.path.join(dataDir, "cleanedSlack/")
minutesLinkDir = os.path.join(dataDir, "minutesLink/")

#website
web_loader = WebBaseLoader(website_urls)
web_docs = web_loader.load()
for d in web_docs:
    d.metadata["source"] = "website"
print(f"Loaded {len(web_docs)} pages from website.")

#google docs
pdf_docs = []
for filename in os.listdir(driveDataDir):
    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(driveDataDir, filename))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = "pdf"
            d.metadata["filename"] = filename
        pdf_docs.extend(docs)
print(f"Loaded {len(pdf_docs)} PDF documents.")

#emails
json_docs = []
if os.path.exists(emailsJson):
    print(f"Loading JSON file: {emailsJson}")
    with open(emailsJson, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, msg in enumerate(data):
        body_text = msg.get("body", "").strip()
        if body_text:  
            json_docs.append(Document(
                page_content=body_text,
                metadata={
                    "source": "email",
                    "from": msg.get("from", ""),
                    "to": msg.get("to", ""),
                    "email_id": i
                }
            ))
    print(f"Loaded {len(json_docs)} email documents.")
else:
    print(f"JSON file not found: {emailsJson}")

#minuteslink
minutes_docs = []
for filename in os.listdir(minutesLinkDir):
    if filename.lower().endswith(".txt"):
        loader = TextLoader(os.path.join(minutesLinkDir, filename), encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = "minutes"
            d.metadata["filename"] = filename
        minutes_docs.extend(docs)
print(f"Loaded {len(minutes_docs)} MinutesLink documents.")

#slack
slack_docs = []
for filename in os.listdir(slackDir):
    if filename.lower().endswith(".txt"):
        with open(os.path.join(slackDir, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if line.strip():
                slack_docs.append(Document(
                    page_content=line.strip(),
                    metadata={"source": "slack", "channel": filename.replace(".txt", "")}
                ))
print(f"Loaded {len(slack_docs)} Slack messages.")

#combine all docs
all_docs = web_docs + pdf_docs + json_docs + minutes_docs + slack_docs
print(f"Total combined documents: {len(all_docs)}")

embeddingModel = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory=persistDir,
    embedding_function=embeddingModel
)
existing = vectorstore.get(include=["metadatas"])
existing_hashes = set()

for md in existing["metadatas"]:
    if md and "hash" in md:
        existing_hashes.add(md["hash"]) 


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(all_docs)

new_chunks = []
for chunk in chunks:
    h = content_hash(chunk.page_content)
    if h not in existing_hashes:
        chunk.metadata["hash"] = h
        new_chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")
print(f"New chunks to embed: {len(new_chunks)}")


batch_size = 512

for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding new content"):
    batch = new_chunks[i:i + batch_size]
    vectorstore.add_documents(batch)

print(f"Stored all embeddings persistently in: {persistDir}")
