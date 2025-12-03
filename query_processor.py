# query_processor.py - Enhanced with semantic analysis
import json
from typing import List, Tuple, Dict, Any
from langchain_core.prompts import PromptTemplate

def detect_query_intent(query: str, llm) -> Dict[str, Any]:
    """
    Analyze query intent semantically, not just keywords.
    Returns intent classification with reasoning.
    """
    intent_prompt = f"""
    Analyze this query and determine what type of response is needed:
    
    Query: {query}
    
    Response Types:
    1. FACTUAL_RETRIEVAL: Answer exists in documents (Who? What? When? Where?)
    2. CREATIVE_GENERATION: Create something new (plans, strategies, documents)
    3. ANALYTICAL_INFERENCE: Analysis, projections, recommendations, comparisons
    4. EXTERNAL_KNOWLEDGE: General knowledge not specific to company
    
    Consider:
    - Questions with "projected", "likely", "potential", "should", "could" → ANALYTICAL_INFERENCE
    - Questions with "create", "draft", "write", "plan" → CREATIVE_GENERATION  
    - Questions with "what are", "who is", "when did" → FACTUAL_RETRIEVAL
    - Questions about other companies/world → EXTERNAL_KNOWLEDGE
    
    Return as JSON:
    {{
        "intent": "FACTUAL_RETRIEVAL|CREATIVE_GENERATION|ANALYTICAL_INFERENCE|EXTERNAL_KNOWLEDGE",
        "reasoning": "Brief explanation",
        "confidence": 0.9,
        "requires_analysis": true/false,
        "requires_generation": true/false
    }}
    
    JSON:
    """
    
    try:
        response = llm.invoke(intent_prompt)
        
        # Try to parse JSON
        content = response.content
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
            
            data = json.loads(json_str)
            return data
            
    except Exception as e:
        print(f"[DEBUG] Intent detection error: {e}")
    
    # Fallback to keyword-based detection
    return fallback_intent_detection(query)

def fallback_intent_detection(query: str) -> Dict[str, Any]:
    """Fallback intent detection using keywords"""
    query_lower = query.lower()
    
    # Analytical inference keywords
    analytical_keywords = [
        "projected", "projection", "forecast", "estimate", "likely", "potential",
        "analysis", "analyze", "infer", "inference", "conclude", "conclusion",
        "recommend", "suggest", "advise", "should", "could", "would", "might",
        "compare", "comparison", "versus", "vs", "contrast", "difference",
        "benefit", "advantage", "disadvantage", "pro", "con", "impact",
        "effect", "result", "outcome", "savings", "cost", "revenue", "profit",
        "efficiency", "improvement", "optimization", "scalability"
    ]
    
    # Creative generation keywords  
    creative_keywords = [
        "create", "develop", "draft", "write", "plan", "design", "propose",
        "generate", "build", "formulate", "brainstorm", "invent", "innovate",
        "action plan", "business plan", "strategy", "roadmap", "framework",
        "template", "outline", "structure", "model", "system", "process"
    ]
    
    # Check for analytical inference
    if any(keyword in query_lower for keyword in analytical_keywords):
        return {
            "intent": "ANALYTICAL_INFERENCE",
            "reasoning": "Contains analytical/inference keywords",
            "confidence": 0.7,
            "requires_analysis": True,
            "requires_generation": True
        }
    
    # Check for creative generation
    if any(keyword in query_lower for keyword in creative_keywords):
        return {
            "intent": "CREATIVE_GENERATION",
            "reasoning": "Contains creative/generation keywords",
            "confidence": 0.7,
            "requires_analysis": False,
            "requires_generation": True
        }
    
    # Default to factual retrieval
    return {
        "intent": "FACTUAL_RETRIEVAL",
        "reasoning": "Default to factual retrieval",
        "confidence": 0.5,
        "requires_analysis": False,
        "requires_generation": False
    }

def is_creative_or_analytical_query(query: str, llm) -> bool:
    """Check if query requires creative generation OR analytical inference"""
    intent = detect_query_intent(query, llm)
    
    # Both creative generation and analytical inference need special handling
    return intent.get("requires_generation", False) or intent.get("requires_analysis", False)

def get_query_intent_details(query: str, llm) -> Dict[str, Any]:
    """Get detailed intent classification"""
    return detect_query_intent(query, llm)

# Keep the other functions but update enhance_query_with_history to use new intent detection
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

# Rest of the functions remain the same...
def decompose_query(query: str, llm) -> Tuple[List[str], List[str]]:
    # ... existing code ...
    pass

def fallback_decomposition(query: str) -> Tuple[List[str], List[str]]:
    # ... existing code ...
    pass

def combine_decomposed_results(query: str, internal_results: List[str], external_results: List[str], llm) -> str:
    # ... existing code ...
    pass