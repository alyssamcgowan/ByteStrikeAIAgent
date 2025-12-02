# search_classifier.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

class SearchDecision(BaseModel):
    """Classification of search type needed for a query"""
    search_type: Literal["internal", "external", "both"] = Field(
        description="Type of search required: 'internal' for company-specific info, 'external' for general knowledge, 'both' for mixed queries"
    )
    reasoning: str = Field(description="Brief explanation for the classification decision")
    # Removed confidence field since LLM struggles with numeric output

class SearchClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.classification_prompt = PromptTemplate(
            template="""You are a search intent classifier for ByteStrike's AI assistant. 
            Determine whether the user's query requires searching internal company documents, external internet sources, or both.

            INTERNAL SEARCH should be used for:
            - ByteStrike company-specific information (strategy, operations, team, communications)
            - Founder's writing style and tone examples
            - Internal processes, job descriptions, or workstreams
            - ByteStrike-specific historical data or documents
            - Drafting emails, investor updates, or communications in founder's voice
            - Questions about ByteStrike's current projects or team
            - Questions using "our", "we", "us" referring to ByteStrike

            EXTERNAL SEARCH should be used for:
            - General knowledge questions (e.g., "Who is the CEO of Nvidia?")
            - Market research not specific to ByteStrike
            - Technology trends and industry news
            - Information about people/companies not directly related to ByteStrike operations
            - Factual information about the world (e.g., capitals, historical facts)
            - Questions that don't mention ByteStrike or internal company context

            BOTH should be used when:
            - Questions that compare ByteStrike to others: "Are there companies similar to Byte-Strike?"
            - Competitive analysis: "How do we compare to [competitor]?"
            - Questions that need both ByteStrike context AND external market data
            - "Research potential CTO candidates for ByteStrike"
            - "Find investors similar to our current ones"
            - Questions about ByteStrike's position in the market

            Query: {query}
            
            Analyze the query and respond with ONLY the classification (internal/external/both) and brief reasoning.
            
            Examples:
            Q: "Who is the CEO of ByteStrike?" → internal
            Q: "Who is the CEO of Nvidia?" → external  
            Q: "Are there companies similar to Byte-Strike?" → both
            Q: "Research potential CTO candidates for ByteStrike" → both
            Q: "What is our current strategy?" → internal
            
            Classification:""",
            input_variables=["query"]
        )
        
        # Combined search reasoning template
        self.combined_reasoning_template = PromptTemplate(
            template="""Analyze this query and determine what aspects require internal vs external search:

            Query: {query}

            Break down:
            1. What specific ByteStrike/internal information is needed?
            2. What external/general knowledge is needed?
            3. How should these be combined in the final answer?
            
            Provide a brief reasoning (1-2 sentences):""",
            input_variables=["query"]
        )
        
        self.structured_llm = self.llm.with_structured_output(SearchDecision)
        self.classification_chain = self.classification_prompt | self.structured_llm
        self.combined_reasoning_chain = self.combined_reasoning_template | self.llm
    
    def classify_search_intent(self, query: str) -> SearchDecision:
        """Classify what type of search is needed for a query"""
        try:
            return self.classification_chain.invoke({"query": query})
        except Exception as e:
            # Fallback if structured output fails
            print(f"Classifier error: {e}")
            return SearchDecision(
                search_type="external",
                reasoning="Default to external due to classification error"
            )
    
    def get_combined_search_reasoning(self, query: str) -> str:
        """Get reasoning for how to combine internal and external search for a query"""
        try:
            return self.combined_reasoning_chain.invoke({"query": query}).content
        except Exception as e:
            return f"Could not generate combined reasoning: {str(e)}"
    
    def should_search_internally(self, query: str) -> bool:
        """Determine if internal search is needed"""
        decision = self.classify_search_intent(query)
        return decision.search_type in ["internal", "both"]
    
    def should_search_externally(self, query: str) -> bool:
        """Determine if external search is needed"""
        decision = self.classify_search_intent(query)
        return decision.search_type in ["external", "both"]
    
    def get_search_plan(self, query: str) -> Tuple[SearchDecision, str]:
        """Get complete search plan including classification and combined reasoning"""
        decision = self.classify_search_intent(query)
        reasoning = ""
        
        if decision.search_type == "both":
            reasoning = self.get_combined_search_reasoning(query)
        
        return decision, reasoning