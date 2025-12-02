# query_processor.py
import json
from typing import List, Tuple

def is_creative_query(query: str) -> bool:
    """Detect if query requires creative generation vs factual retrieval"""
    creative_keywords = [
        "create", "develop", "draft", "write", "plan", "design", "propose",
        "suggest", "recommend", "outline", "generate", "build", "formulate",
        "brainstorm", "what should", "how can", "how would", "next steps",
        "action plan", "business plan", "strategy", "roadmap", "improve",
        "enhance", "optimize", "new approach", "innovative"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in creative_keywords)

def enhance_query_with_history(query: str, history_context: str, llm) -> str:
    """Enhance query with chat history context"""
    if not history_context:
        return query
    
    # Check if query is ambiguous without context
    ambiguous_indicators = ["yes", "no", "sure", "ok", "okay", "that", "it", "them", "those"]
    query_lower = query.lower().strip().rstrip('!?.')
    
    if query_lower in ambiguous_indicators or len(query.split()) <= 2:
        # This is a follow-up, need to understand from context
        enhancement_prompt = f"""
        Chat History:
        {history_context}
        
        Current ambiguous user message: "{query}"
        
        Based on the conversation history, what is the user most likely referring to or asking about?
        Rephrase this as a clear, complete question.
        
        Clear question:
        """
        
        try:
            response = llm.invoke(enhancement_prompt)
            enhanced = response.content.strip()
            print(f"[DEBUG] Enhanced ambiguous query: '{query}' -> '{enhanced}'")
            return enhanced
        except Exception as e:
            print(f"[DEBUG] Query enhancement failed: {e}")
            return query
    
    return query

def decompose_query(query: str, llm) -> Tuple[List[str], List[str]]:
    """
    Decompose a complex query into internal and external parts.
    Returns: (internal_queries, external_queries)
    """
    decomposition_prompt = f"""
    Analyze this query and separate it into:
    1. Parts that require ByteStrike internal document search
    2. Parts that require general knowledge/external search
    
    Query: {query}
    
    Guidelines:
    - ByteStrike internal: Anything about ByteStrike company, business, strategy, team, documents
    - External: General knowledge, facts about other companies, places, people not related to ByteStrike
    
    Return as JSON with two arrays:
    {{
        "internal": ["list of internal search queries"],
        "external": ["list of external search queries"]
    }}
    
    JSON:
    """
    
    try:
        response = llm.invoke(decomposition_prompt)
        
        # Try to parse JSON
        content = response.content
        if "{" in content and "}" in content:
            # Extract JSON part
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
            
            data = json.loads(json_str)
            internal_queries = data.get("internal", [])
            external_queries = data.get("external", [])
            
            print(f"[DEBUG] Query decomposed:")
            print(f"  Internal: {internal_queries}")
            print(f"  External: {external_queries}")
            
            return internal_queries, external_queries
            
    except Exception as e:
        print(f"[DEBUG] Decomposition error: {e}")
    
    # Fallback: simple keyword-based decomposition
    return fallback_decomposition(query)

def fallback_decomposition(query: str) -> Tuple[List[str], List[str]]:
    """Fallback decomposition when AI decomposition fails"""
    query_lower = query.lower()
    
    # Simple rule: if contains ByteStrike keywords, it's internal
    if any(keyword in query_lower for keyword in ["bytestrike", "byte strike", "our company", "our business"]):
        return [query], []
    else:
        return [], [query]

def combine_decomposed_results(query: str, internal_results: List[str], external_results: List[str], llm) -> str:
    """Combine results from decomposed queries"""
    if not internal_results and not external_results:
        return "I couldn't find information to answer your query."
    
    combine_prompt = f"""
    Original query: {query}
    
    {'ByteStrike Information:' + chr(10) + chr(10).join(internal_results) if internal_results else ''}
    
    {'External Information:' + chr(10) + chr(10).join(external_results) if external_results else ''}
    
    Provide a comprehensive answer that addresses all parts of the original query.
    Structure it naturally without saying "internal" or "external".
    
    Answer:
    """
    
    try:
        response = llm.invoke(combine_prompt)
        return response.content
    except Exception as e:
        print(f"[DEBUG] Combination error: {e}")
        # Simple fallback
        parts = []
        if internal_results:
            parts.append("\n".join(internal_results))
        if external_results:
            parts.append("\n".join(external_results))
        return "\n\n".join(parts)