# agent.py - Main entry point
import langchain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Import modules
from chat_history import ChatHistory
from query_processor import (
    is_creative_query, 
    enhance_query_with_history,
    decompose_query,
    combine_decomposed_results
)
from search_handlers import create_retrieval_chain, ExternalSearcher
from search_classifier import SearchClassifier

load_dotenv()

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key
)

# Initialize vector store
persistDir = "chroma_store/"
embeddingModel = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

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

# Templates
retrieval_template = """
You are ByteStrike's internal AI assistant. Use ONLY the internal context to answer ByteStrike-specific questions.
If the context doesn't contain relevant information, respond with: "The available documents do not contain that information."
DO NOT use any external knowledge or make up information.

Context:
{context}

Question:
{question}

Answer based ONLY on the context above:
"""

generation_template = """
You are ByteStrike's AI assistant. Your role is to help create plans, strategies, and documents for the company.

Use the ByteStrike context to understand:
- The company's business model, products, and goals
- Current strategy and operations  
- Founder's tone and writing style
- Industry context

Then CREATE a new plan/document/strategy based on the user's request.

ByteStrike Context:
{context}

User Request:
{question}

Based on the ByteStrike context above, create what was requested. Be specific, actionable, and professional.
If certain information is missing from context, use your best judgment to fill reasonable gaps.

Response:
"""

# Initialize components
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retrieval_chain = create_retrieval_chain(retriever, retrieval_template, llm)
generation_chain = create_retrieval_chain(retriever, generation_template, llm)
classifier = SearchClassifier()
chat_history = ChatHistory()
external_searcher = ExternalSearcher()

def handle_both_decomposed(query: str) -> str:
    """Handle 'both' classification with query decomposition"""
    print(f"[DEBUG] Decomposing query for separate searches")
    
    # Step 1: Decompose query
    internal_queries, external_queries = decompose_query(query, llm)
    
    # Step 2: Execute internal searches
    internal_results = []
    for internal_query in internal_queries:
        try:
            result = retrieval_chain.invoke({"query": internal_query})
            if "do not contain that information" not in result["result"].lower():
                internal_results.append(result["result"])
        except Exception as e:
            print(f"[DEBUG] Internal search error for '{internal_query}': {e}")
    
    # Step 3: Execute external searches
    external_results = []
    for external_query in external_queries:
        result = external_searcher.search(external_query)
        if result and "couldn't find" not in result.lower():
            external_results.append(result)
    
    # Step 4: Combine results
    return combine_decomposed_results(query, internal_results, external_results, llm)

def handle_creative_query(query: str) -> str:
    """Handle creative/planning queries that need generation"""
    print(f"[DEBUG] Handling creative query: '{query}'")
    
    # Use generation chain which allows creative output
    result = generation_chain.invoke({"query": query})
    return result["result"]

def agent(query: str, user_id: str = "default") -> str:
    """Run a query against the ByteStrike RAG system and return the result."""
    # Get chat history context
    history_context = chat_history.get_context(user_id)
    
    # Enhance query with chat history context
    enhanced_query = enhance_query_with_history(query, history_context, llm)
    
    # Add user message to history
    chat_history.add_message(user_id, "user", query)
    
    # Check if this is a creative query first
    if is_creative_query(enhanced_query):
        print(f"[DEBUG] Creative query detected, using generation")
        response = handle_creative_query(enhanced_query)
    else:
        # Classify search intent for factual queries
        search_decision = classifier.classify_search_intent(enhanced_query)
        
        # DEBUG: Show classification
        print(f"[DEBUG] Query: '{query}' -> Enhanced: '{enhanced_query}' -> {search_decision.search_type}")
        print(f"[DEBUG] Reasoning: {search_decision.reasoning}")
        
        # Route based on classification
        if search_decision.search_type == "internal":
            print(f"[DEBUG] Using internal RAG only")
            result = retrieval_chain.invoke({"query": enhanced_query})
            response = result["result"]
            
        elif search_decision.search_type == "external":
            print(f"[DEBUG] Using external search only")
            response = external_searcher.search(enhanced_query)
            
        else:  # both - Use decomposition
            print(f"[DEBUG] Using decomposition for 'both' query")
            response = handle_both_decomposed(enhanced_query)
    
    # Add assistant response to history
    chat_history.add_message(user_id, "assistant", response)
    
    return response

if __name__ == "__main__":
    # simple manual test
    test_query = input("What do you want to search? ")
    print(agent(test_query))