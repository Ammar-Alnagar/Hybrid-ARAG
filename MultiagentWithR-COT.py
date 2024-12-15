import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain and Core Imports
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Agent and Tool Imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.memory import ConversationBufferMemory

class DocumentRetrievalAgent(BaseTool):
    """Specialized agent for document retrieval"""
    name = "document_retrieval_agent"
    description = "Advanced document retrieval with contextual analysis"

    def __init__(self, retriever, llm):
        super().__init__()
        self.retriever = retriever
        self.llm = llm

    def _run(
        self, 
        query: str, 
        run_manager: CallbackManagerForToolRun = None
    ) -> str:
        """Execute advanced document retrieval"""
        try:
            # Retrieve documents
            docs = self.retriever.get_relevant_documents(query)

            # Advanced context analysis
            analysis_prompt = PromptTemplate.from_template("""
            Perform a comprehensive context analysis for the query: {query}
            
            Retrieved Documents:
            {documents}
            
            Context Analysis:
            1. Identify key themes and relationships
            2. Assess document relevance
            3. Extract critical insights
            4. Summarize contextual understanding
            """)

            # Format documents
            formatted_docs = "\n\n".join([
                f"Document {i+1} (Relevance Score: {doc.metadata.get('relevance_score', 'N/A')}):\n{doc.page_content}"
                for i, doc in enumerate(docs)
            ])

            # Analyze context
            analysis_input = analysis_prompt.format(
                query=query, 
                documents=formatted_docs
            )
            context_analysis = self.llm.invoke(analysis_input).content

            return context_analysis

        except Exception as e:
            return f"Retrieval Error: {str(e)}"

    async def _arun(self, query: str):
        return self._run(query)

class QueryRewriteAgent(BaseTool):
    """Advanced query rewriting agent"""
    name = "query_rewrite_agent"
    description = "Sophisticated query reformulation and expansion"

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def _run(
        self, 
        query: str, 
        run_manager: CallbackManagerForToolRun = None
    ) -> str:
        """Rewrite and expand query"""
        rewrite_prompt = PromptTemplate.from_template("""
        Advanced Query Reformulation Process:
        
        Original Query: {query}
        
        Reformulation Steps:
        1. Decompose the query into core semantic components
        2. Identify implicit information needs
        3. Generate semantically related terms
        4. Reconstruct a more precise, comprehensive query
        
        Reformulated Query:""")

        # Invoke LLM for query rewriting
        prompt = rewrite_prompt.format(query=query)
        reformulated_query = self.llm.invoke(prompt).content

        return reformulated_query

    async def _arun(self, query: str):
        return self._run(query)

class ReasoningAgent(BaseTool):
    """Advanced reasoning and synthesis agent"""
    name = "reasoning_agent"
    description = "Comprehensive reasoning and answer synthesis"

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def _run(
        self, 
        query: str, 
        context: str, 
        run_manager: CallbackManagerForToolRun = None
    ) -> str:
        """Perform reasoning and synthesize answer"""
        reasoning_prompt = PromptTemplate.from_template("""
        Advanced Reasoning Framework:
        
        Query: {query}
        Context: {context}
        
        Reasoning Process:
        1. Analyze query intent
        2. Map context to query requirements
        3. Identify logical connections
        4. Synthesize comprehensive response
        5. Validate reasoning chain
        
        Reasoned Response:""")

        # Synthesize reasoned response
        prompt = reasoning_prompt.format(query=query, context=context)
        reasoned_response = self.llm.invoke(prompt).content

        return reasoned_response

    async def _arun(self, query: str, context: str):
        return self._run(query, context)

class ProofreadingAgent(BaseTool):
    """Comprehensive proofreading and validation agent"""
    name = "proofreading_agent"
    description = "Linguistic and factual validation"

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def _run(
        self, 
        text: str, 
        run_manager: CallbackManagerForToolRun = None
    ) -> str:
        """Proofread and validate text"""
        proofreading_prompt = PromptTemplate.from_template("""
        Comprehensive Proofreading Checklist:
        
        Text to Proofread:
        {text}
        
        Proofreading Criteria:
        1. Grammar and syntax correctness
        2. Coherence and logical flow
        3. Factual accuracy
        4. Language clarity and precision
        5. Potential bias or ambiguity
        
        Proofread and Validated Text:""")

        # Invoke proofreading
        prompt = proofreading_prompt.format(text=text)
        proofread_result = self.llm.invoke(prompt).content

        return proofread_result

    async def _arun(self, text: str):
        return self._run(text)

class RecursiveChainOfThoughtAgent(BaseTool):
    """Advanced Recursive Chain of Thought Agent"""
    name = "recursive_chain_of_thought_agent"
    description = "Hierarchical problem decomposition and recursive reasoning"

    def __init__(self, llm, max_depth=3):
        super().__init__()
        self.llm = llm
        self.max_depth = max_depth

    def _run(
        self, 
        query: str, 
        context: str = None,
        depth: int = 0,
        run_manager: CallbackManagerForToolRun = None
    ) -> str:
        """Recursive reasoning with hierarchical decomposition"""
        if depth >= self.max_depth:
            return "Maximum reasoning depth reached."

        recursive_cot_prompt = PromptTemplate.from_template("""
        Recursive Chain of Thought Reasoning Framework:

        Current Problem Depth: {depth}
        Problem: {query}
        Context: {context}

        Reasoning Steps:
        1. Problem Decomposition
        2. Identify Sub-Problems
        3. Recursive Analysis
        4. Solution Synthesis
        5. Hierarchical Integration

        Recursive Reasoning Output:""")

        # Prepare context (use empty string if None)
        context = context or ""

        # Invoke LLM for recursive reasoning
        prompt = recursive_cot_prompt.format(
            query=query, 
            context=context, 
            depth=depth
        )
        reasoning_output = self.llm.invoke(prompt).content

        # Check if further decomposition is needed
        decomposition_check_prompt = PromptTemplate.from_template("""
        Assess the complexity of the current reasoning output:
        
        Reasoning Output: {reasoning_output}
        Original Problem: {query}

        Complexity Assessment:
        1. Determine if problem requires further decomposition
        2. Identify potential sub-problems
        3. Recommend recursive depth

        Assessment:""")

        complexity_check = self.llm.invoke(
            decomposition_check_prompt.format(
                reasoning_output=reasoning_output, 
                query=query
            )
        ).content

        # Recursive sub-problem handling
        if "further decomposition" in complexity_check.lower():
            # Extract potential sub-problems
            sub_problems = re.findall(r'Sub-Problem\s*\d+:\s*(.+)', complexity_check)
            
            sub_problem_solutions = []
            for sub_problem in sub_problems:
                sub_solution = self._run(
                    query=sub_problem, 
                    context=reasoning_output, 
                    depth=depth + 1
                )
                sub_problem_solutions.append(sub_solution)

            # Integrate sub-problem solutions
            integration_prompt = PromptTemplate.from_template("""
            Solution Integration Framework:
            
            Original Problem: {query}
            Sub-Problem Solutions:
            {sub_solutions}

            Integration Steps:
            1. Analyze individual solutions
            2. Identify common threads
            3. Synthesize comprehensive solution
            4. Validate holistic approach

            Integrated Solution:""")

            integrated_solution = self.llm.invoke(
                integration_prompt.format(
                    query=query, 
                    sub_solutions="\n".join(sub_problem_solutions)
                )
            ).content

            return integrated_solution

        return reasoning_output

    async def _arun(self, query: str, context: str = None):
        return self._run(query, context)

def create_multi_agent_rag_system(
    embeddings, 
    retriever, 
    llm, 
    verbose: bool = True
):
    """Create a multi-agent RAG system with Recursive Chain of Thought"""
    # Initialize specialized agents
    document_retrieval_agent = DocumentRetrievalAgent(retriever, llm)
    query_rewrite_agent = QueryRewriteAgent(llm)
    reasoning_agent = ReasoningAgent(llm)
    proofreading_agent = ProofreadingAgent(llm)
    recursive_cot_agent = RecursiveChainOfThoughtAgent(llm)

    # Comprehensive RAG workflow
    def multi_agent_rag_workflow(query: str) -> str:
        # Step 1: Query Rewriting
        rewritten_query = query_rewrite_agent._run(query)
        print("🔍 Rewritten Query:", rewritten_query)

        # Step 2: Document Retrieval
        retrieved_context = document_retrieval_agent._run(rewritten_query)
        print("📚 Retrieved Context Summary:\n", retrieved_context)

        # Step 3: Recursive Chain of Thought
        recursive_reasoning = recursive_cot_agent._run(
            query=rewritten_query, 
            context=retrieved_context
        )
        print("🌳 Recursive Reasoning:\n", recursive_reasoning)

        # Step 4: Traditional Reasoning and Synthesis
        reasoned_response = reasoning_agent._run(rewritten_query, recursive_reasoning)
        print("🧠 Reasoned Response:\n", reasoned_response)

        # Step 5: Proofreading and Validation
        final_response = proofreading_agent._run(reasoned_response)
        print("✅ Final Validated Response:\n", final_response)

        return final_response

    return multi_agent_rag_workflow

def initialize_multi_agent_rag_system():
    """Initialize the multi-agent RAG system"""
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

    # Create Multi-Agent RAG System
    multi_agent_rag = create_multi_agent_rag_system(
        embeddings, 
        retriever, 
        llm, 
        verbose=True
    )

    return multi_agent_rag

def main():
    """Main interaction loop for Multi-Agent RAG System"""
    print("\n🤖 مرحباً! أنا نظام متعدد الوكلاء للذكاء الاصطناعي.")

    # Initialize Multi-Agent RAG system
    multi_agent_rag = initialize_multi_agent_rag_system()

    while True:
        user_question = input("\nأدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")

        if user_question.lower() in ['quit', 'خروج']:
            break

        try:
            # Run multi-agent workflow
            final_response = multi_agent_rag(user_question)

            print("\n📝 الإجابة النهائية:")
            print(final_response)

        except Exception as e:
            print(f"\n❌ حدث خطأ: {str(e)}")

    print("\nشكراً لاستخدامك النظام متعدد الوكلاء. إلى اللقاء!")

if __name__ == "__main__":
    main()