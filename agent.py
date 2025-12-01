# agent.py (ASYNC STREAMING VERSION)
import os
from typing import AsyncGenerator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_pipeline import retrieve_docs  # async version of your RAG pipeline

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------------------
# Load LLM once (supports async calls)
# Using gpt-3.5-turbo as it's efficient for RAG streaming
# -------------------------------------
llm = ChatOpenAI(
    model="gpt-5-nano", 
    temperature=0,
    api_key=OPENAI_KEY
)

template = """
You are ByteStrike’s internal Slack AI assistant, supporting the founder in growing and managing the company.
Assume any queries brought to you refer to the company ByteStrike.

Your responsibilities include:
Drafting professional communications (emails, investor updates, outreach messages, etc.) in the founder’s tone and writing style, based on the examples in the internal document store.
Creating job descriptions, team workstreams, and project outlines.
Researching and summarizing information about potential hires (e.g., CTO candidates), investors (VCs), competitors, and market/technology trends.
Helping organize company operations, strategy, and internal documentation efficiently.

Instructions:
Use the context from ByteStrike’s internal document vector store to inform answers and emulate the founder’s style.
Do not fabricate company-specific facts not present in documents.
If uncertain, say: “The available documents do not contain that information.”

Be concise, accurate, and reflective of the founder’s writing style.

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


# -------------------------------------
# Build the LangChain RAG Chain
# -------------------------------------

# 1. Define the Context Formatter
# This function prepares the context from the list of documents
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 2. Define the main chain structure
# We use RunnablePassthrough to pass the 'question' key through to the prompt
# while using the 'context' from the retrieval step.
rag_chain = (
    # This part prepares the inputs for the final prompt
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x["documents"])
    )
    | prompt  # Apply the prompt template
    | llm     # Call the LLM
)


# -------------------------------------
# ASYNC STREAMING Agent Function
# -------------------------------------
async def stream_agent(query: str) -> AsyncGenerator[str, None]:
    print("\n==============================", flush=True)
    print(f"AGENT RECEIVED QUERY: {query}", flush=True)
    print("Starting retrieval...", flush=True)
    print("==============================", flush=True)

    # 1. Retrieve docs async
    # The 'retrieve_docs' function is assumed to return a List[Document]
    docs = await retrieve_docs(query)

    print(f"Retrieved {len(docs)} documents.", flush=True)

    if not docs:
        print("No docs found — returning fallback message.", flush=True)
        # If no documents are found, immediately yield a fallback message
        yield "No relevant ByteStrike documents were found for this question."
        return # Exit the generator

    # Prepare the input dictionary for the LangChain Runnable
    chain_input = {
        "question": query,
        # The rag_chain expects the raw documents here for the formatter
        "documents": docs 
    }

    print("Starting LLM streaming...", flush=True)

    # 2. Call the chain asynchronously using astream()
    # astream() returns an async generator
    async for chunk in rag_chain.astream(chain_input):
        # LangChain's astream() yields chunks, usually MessageChunks or string content.
        # We only care about the final output of the chain, which is the LLM's content.
        
        # When streaming the final output, the chunk will be a BaseMessageChunk 
        # (or similar object) that contains the text fragment in its 'content' attribute.
        if hasattr(chunk, 'content') and chunk.content is not None:
            yield chunk.content
            
    print("LLM streaming complete.", flush=True)
    print("==============================\n", flush=True)