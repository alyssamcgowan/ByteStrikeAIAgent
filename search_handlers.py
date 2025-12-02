# search_handlers.py
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from tavily import TavilyClient
import os

def create_retrieval_chain(retriever, template: str, llm):
    """Create a RAG chain with given template"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff", 
        chain_type_kwargs={"prompt": prompt}
    )

class ExternalSearcher:
    def __init__(self):
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
    
    def search(self, query: str) -> str:
        """Search external sources using Tavily API"""
        if not self.tavily:
            return "External search not configured."
            
        try:
            response = self.tavily.search(
                query=query,
                search_depth="basic",
                max_results=3,
                include_answer=True,
                include_raw_content=False
            )
            
            if response.get("answer"):
                return response["answer"]
            
            if response.get("results"):
                # Take the first good result
                for res in response["results"]:
                    if res.get("content"):
                        return res["content"][:500]
            
            return f"I couldn't find information about '{query}'."
            
        except Exception as e:
            return f"Search unavailable. Please try again later."