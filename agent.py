# agent.py - Main entry point with analytical inference
import langchain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Import modules
from chat_history import ChatHistory
from query_processor import (
    is_creative_or_analytical_query,  # UPDATED: New function name
    enhance_query_with_history,
    decompose_query,
    combine_decomposed_results,
    get_query_intent_details  # NEW: Get detailed intent classification
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

# ---------- TEMPLATES ----------
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
You are ByteStrike's AI assistant. Your role is to CREATE new plans, documents, and strategies.

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

Based on the ByteStrike context above, CREATE what was requested. Be specific, actionable, and professional.
Focus on generating NEW content, not just summarizing.

Response:
"""

# NEW: Analytical inference template
analytical_template = """
You are ByteStrike's AI assistant. Your role is to provide analysis, projections, and inferences based on ByteStrike's context.

Use the ByteStrike context to understand:
- The company's business model, products, and services
- Current operations, technology, and market position
- Relevant data points and metrics
- Industry trends and competitive landscape

Then ANALYZE and provide informed projections/recommendations based on the user's request.

ByteStrike Context:
{context}

User Request:
{question}

Based on the ByteStrike context above, provide an analytical response that:
1. Makes logical inferences from available information
2. Provides reasoned projections or estimates where appropriate
3. Acknowledges uncertainties or missing data
4. Offers practical recommendations or insights
5. Uses quantitative estimates when possible (e.g., "likely 20-30%", "potentially $X in savings")

Response:
"""

# Initialize components
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create three chains: retrieval, generation, and analytical
retrieval_chain = create_retrieval_chain(retriever, retrieval_template, llm)
generation_chain = create_retrieval_chain(retriever, generation_template, llm)
analytical_chain = create_retrieval_chain(retriever, analytical_template, llm)  # NEW

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

def handle_creative_or_analytical_query(query: str, intent_details: dict) -> str:
    """Handle creative generation OR analytical inference queries"""
    print(f"[DEBUG] Handling {'analytical' if intent_details.get('requires_analysis') else 'creative'} query")
    
    if intent_details.get("requires_analysis", False):
        # Use analytical chain for projections, inferences, recommendations
        result = analytical_chain.invoke({"query": query})
    else:
        # Use generation chain for creation tasks
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
    
    # Get detailed intent classification
    intent_details = get_query_intent_details(enhanced_query, llm)
    print(f"[DEBUG] Query intent: {intent_details['intent']} (confidence: {intent_details['confidence']:.2f})")
    
    # Check if this needs special handling (creative OR analytical)
    if intent_details.get("requires_generation", False) or intent_details.get("requires_analysis", False):
        print(f"[DEBUG] Using {'analytical' if intent_details.get('requires_analysis') else 'creative'} chain")
        response = handle_creative_or_analytical_query(enhanced_query, intent_details)
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