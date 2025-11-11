import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from embed_and_store import setup_vector_store

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the chat model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

def search_documents(query, k=3):
    """Search for relevant documents using the vector store"""
    vectorstore = setup_vector_store()
    results = vectorstore.similarity_search(query, k=k)
    return results

def research_with_context(query):
    """Perform research using both the model and stored documents"""
    
    # Search for relevant documents
    relevant_docs = search_documents(query)
    
    # Build context from documents
    context = ""
    if relevant_docs:
        context = "Relevant information from research:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"Source {i}:\n{doc.page_content}\n\n"
    
    # Create the enhanced prompt
    enhanced_prompt = f"""
    You are a research assistant. Use the following context to answer the user's question.
    If the context doesn't contain relevant information, use your general knowledge.
    
    {context}
    
    User Question: {query}
    
    Please provide a comprehensive answer based on the available information.
    """
    
    # Get response from the model
    response = model.invoke(enhanced_prompt)
    return response.content

def main():
    # Setup vector store (will load existing or create new)
    vectorstore = setup_vector_store()
    
    query = input("What do you want to research? ")
    
    response = research_with_context(query)
    
    print("\n" + "="*50)
    print("RESEARCH RESULTS:")
    print("="*50)
    print(response)

if __name__ == "__main__":
    main()