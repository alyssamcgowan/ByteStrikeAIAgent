# agent.py (ASYNC VERSION WITH LOGGING)
import os
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from rag_pipeline import retrieve_docs  # async version of your RAG pipeline

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------------------
# Load LLM once (supports async calls)
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
# ASYNC Agent Function
# -------------------------------------
async def agent(query: str) -> str:
    print("\n==============================")
    print(f"AGENT RECEIVED QUERY: {query}")
    print("Starting retrieval...")
    print("==============================")

    # 1. Retrieve docs async
    docs = await retrieve_docs(query)

    print(f"Retrieved {len(docs)} documents.")

    if not docs:
        print("No docs found — returning fallback message.")
        return "No relevant ByteStrike documents were found for this question."

    context = "\n\n".join(d.page_content for d in docs)

    # 2. Construct final prompt
    final_prompt = prompt.format(context=context, question=query)

    print("Sending prompt to LLM...")

    # 3. Call LLM asynchronously
    response = await llm.ainvoke(final_prompt)

    print("LLM response received.")
    print("==============================\n")

    return response.content
