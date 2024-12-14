import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain and Embedding Imports
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

# LangChain Agent and Tool Imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Additional Imports
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForToolRun

class DocumentRetrievalTool(BaseTool):
    """Custom tool for retrieving relevant documents"""
    name = "document_retrieval"
    description = "Useful for retrieving relevant documents based on a query"
    
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
    
    def _run(
        self, 
        query: str, 
        run_manager: CallbackManagerForToolRun = None
    ) -> str:
        """Execute document retrieval"""
        try:
            # Retrieve top 5 most relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Format retrieved documents
            formatted_docs = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])
            
            return formatted_docs
        except Exception as e:
            return f"Error in document retrieval: {str(e)}"
    
    async def _arun(self, query: str):
        """Async version of document retrieval"""
        return self._run(query)

class QueryReformulationTool(BaseTool):
    """Tool for reformulating queries"""
    name = "query_reformulation"
    description = "Reformulates user queries to improve search precision"
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    
    def _run(
        self, 
        query: str, 
        run_manager: CallbackManagerForToolRun = None
    ) -> str:
        """Reformulate query"""
        reformulation_prompt = PromptTemplate.from_template(
            """You are an expert query reformulation assistant. 
            Reformulate the following query to be more precise and searchable:
            
            Original Query: {query}
            
            Reformulated Query:"""
        )
        
        prompt = reformulation_prompt.format(query=query)
        reformulated_query = self.llm.invoke(prompt).content
        
        return reformulated_query
    
    async def _arun(self, query: str):
        """Async version of query reformulation"""
        return self._run(query)

def create_rarag_agent(
    embeddings, 
    retriever, 
    llm, 
    verbose: bool = True
):
    """Create a RARAG Agent with advanced reasoning capabilities"""
    # Initialize custom tools
    document_tool = DocumentRetrievalTool(retriever)
    query_reformulation_tool = QueryReformulationTool(llm)
    
    # Define tools list
    tools = [
        document_tool, 
        query_reformulation_tool
    ]
    
    # Agent prompt with explicit reasoning steps
    agent_prompt = PromptTemplate.from_template("""
    You are an advanced reasoning agent designed to provide comprehensive answers.

    REASONING FRAMEWORK:
    1. Query Analysis
       - Understand the underlying intent of the query
       - Identify key information needs
    
    2. Information Retrieval Strategy
       - Use document retrieval tool to gather relevant context
       - Reformulate query if initial retrieval is insufficient
    
    3. Synthesis and Reasoning
       - Analyze retrieved documents
       - Extract key insights
       - Construct a coherent, well-reasoned response

    4. Response Generation
       - Provide a clear, structured answer
       - Cite sources when possible
       - Maintain academic rigor

    Available Tools: {tool_names}
    Previous Conversation: {chat_history}
    Current Query: {input}

    Thoughts and Reasoning Process:
    {agent_scratchpad}
    """)

    # Create conversational memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    # Create the agent
    agent = create_react_agent(
        llm=llm, 
        tools=tools, 
        prompt=agent_prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=verbose,
        memory=memory,
        max_iterations=5
    )
    
    return agent_executor

def initialize_rarag_system():
    """Initialize the entire RARAG system"""
    # Load environment variables
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")

    # Create embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text", 
        show_progress=False
    )

    # Initialize vector store
    db = Chroma(
        persist_directory="./db-mawared",
        embedding_function=embeddings
    )

    # Create retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # Create RARAG Agent
    rarag_agent = create_rarag_agent(
        embeddings, 
        retriever, 
        llm, 
        verbose=True
    )

    return rarag_agent

def main():
    """Main interaction loop"""
    print("\nمرحباً! أنا مساعدك الذكي باستخدام وكيل متقدم للبحث والتفكير.")
    
    # Initialize RARAG system
    rarag_agent = initialize_rarag_system()
    
    while True:
        user_question = input("\nأدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")
        
        if user_question.lower() in ['quit', 'خروج']:
            break
        
        try:
            # Run agent with user question
            response = rarag_agent.invoke({"input": user_question})
            
            print("\nالإجابة:")
            print(response['output'])
        
        except Exception as e:
            print(f"\nحدث خطأ: {str(e)}")
    
    print("\nشكراً لاستخدامك المساعد الذكي. إلى اللقاء!")

if __name__ == "__main__":
    main()