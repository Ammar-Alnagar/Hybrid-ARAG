import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# LangChain and Core Imports
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Error handling and logging
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class QueryResult:
    """Structured container for query results"""
    original_query: str
    rewritten_query: str
    retrieved_documents: List[Document]
    context_analysis: str
    reasoned_response: str
    final_response: str
    metadata: Dict[str, Any]
    timestamp: datetime

class DocumentRetrievalAgent(Tool):
    """Enhanced document retrieval agent with advanced features"""
    name = "document_retrieval_agent"
    description = "Advanced document retrieval with contextual analysis"

    def __init__(self, retriever, llm, chunk_size: int = 1000):
        super().__init__(name=self.name, func=self._run, description=self.description)
        self.retriever = retriever
        self.llm = llm
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run(self, query: str) -> Dict[str, Any]:
        """Execute enhanced document retrieval with error handling"""
        try:
            # Enhanced document retrieval with metadata
            docs = self.retriever.get_relevant_documents(query)
            
            # Document chunking for better processing
            chunked_docs = self._chunk_documents(docs)
            
            # Advanced context analysis with improved prompt
            analysis_prompt = PromptTemplate.from_template("""
            Perform an advanced contextual analysis for the query:
            Query: {query}
            
            Retrieved Documents:
            {documents}
            
            Required Analysis:
            1. Key Themes and Concepts:
               - Main topics and their relationships
               - Secondary themes and connections
            
            2. Relevance Assessment:
               - Document-query alignment score
               - Information completeness
               - Potential knowledge gaps
            
            3. Critical Insights:
               - Primary findings
               - Supporting evidence
               - Contradictions or uncertainties
            
            4. Contextual Understanding:
               - Historical context
               - Current implications
               - Future considerations
            
            Please provide a structured analysis addressing each point above.
            """)

            # Enhanced document formatting with metadata
            formatted_docs = self._format_documents(chunked_docs)
            
            # Generate comprehensive analysis
            analysis_input = analysis_prompt.format(
                query=query,
                documents=formatted_docs
            )
            
            context_analysis = self.llm.invoke(analysis_input).content
            
            return {
                "documents": docs,
                "analysis": context_analysis,
                "metadata": {
                    "retrieval_timestamp": datetime.now().isoformat(),
                    "doc_count": len(docs),
                    "chunk_count": len(chunked_docs)
                }
            }

        except Exception as e:
            self.logger.error(f"Document retrieval error: {str(e)}")
            raise

    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into manageable chunks"""
        chunked_docs = []
        for doc in docs:
            text = doc.page_content
            chunks = [text[i:i + self.chunk_size] 
                     for i in range(0, len(text), self.chunk_size)]
            
            for i, chunk in enumerate(chunks):
                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                ))
        return chunked_docs

    def _format_documents(self, docs: List[Document]) -> str:
        """Format documents with enhanced metadata"""
        formatted_docs = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            relevance_score = metadata.get('relevance_score', 'N/A')
            chunk_info = f"Chunk {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
            
            formatted_docs.append(
                f"Document {i+1}\n"
                f"Relevance Score: {relevance_score}\n"
                f"Chunk Info: {chunk_info}\n"
                f"Content:\n{doc.page_content}\n"
            )
        return "\n\n".join(formatted_docs)

class QueryRewriteAgent(Tool):
    """Enhanced query rewriting agent with semantic analysis"""
    name = "query_rewrite_agent"
    description = "Advanced query reformulation with semantic analysis"

    def __init__(self, llm):
        super().__init__(name=self.name, func=self._run, description=self.description)
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run(self, query: str) -> Dict[str, Any]:
        """Enhanced query rewriting with semantic analysis"""
        try:
            rewrite_prompt = PromptTemplate.from_template("""
            Advanced Query Analysis and Reformulation:
            
            Original Query: {query}
            
            Process Steps:
            1. Semantic Analysis:
               - Core concepts identification
               - Intent classification
               - Context requirements
            
            2. Query Expansion:
               - Synonym generation
               - Related concepts
               - Temporal considerations
            
            3. Reformulation:
               - Precision enhancement
               - Ambiguity resolution
               - Coverage optimization
            
            4. Validation:
               - Semantic preservation
               - Scope verification
               - Bias checking
            
            Please provide:
            1. Reformulated query
            2. Semantic analysis summary
            3. Expansion rationale
            """)

            prompt = rewrite_prompt.format(query=query)
            result = self.llm.invoke(prompt).content
            
            # Parse structured response
            sections = result.split("\n\n")
            reformulated_query = sections[0].strip()
            
            return {
                "original_query": query,
                "reformulated_query": reformulated_query,
                "analysis": "\n\n".join(sections[1:]),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "semantic_complexity": len(query.split())
                }
            }

        except Exception as e:
            self.logger.error(f"Query rewrite error: {str(e)}")
            raise

class ReasoningAgent(Tool):
    """Enhanced reasoning agent with structured analysis"""
    name = "reasoning_agent"
    description = "Advanced reasoning and synthesis with structured analysis"

    def __init__(self, llm):
        super().__init__(name=self.name, func=self._run, description=self.description)
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run(self, query: str, context: str) -> Dict[str, Any]:
        """Enhanced reasoning with structured analysis"""
        try:
            reasoning_prompt = PromptTemplate.from_template("""
            Advanced Reasoning Framework:
            
            Query: {query}
            Context: {context}
            
            Analysis Requirements:
            1. Query Intent Analysis:
               - Primary objective
               - Secondary requirements
               - Implicit needs
            
            2. Context Mapping:
               - Evidence alignment
               - Information sufficiency
               - Knowledge gaps
            
            3. Logical Framework:
               - Core arguments
               - Supporting evidence
               - Counter-considerations
            
            4. Response Synthesis:
               - Key findings
               - Supporting details
               - Confidence assessment
            
            5. Validation:
               - Logical consistency
               - Evidence support
               - Completeness check
            
            Please provide a comprehensive response addressing all components.
            """)

            prompt = reasoning_prompt.format(query=query, context=context)
            reasoned_response = self.llm.invoke(prompt).content
            
            return {
                "query": query,
                "reasoned_response": reasoned_response,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "context_length": len(context),
                    "response_length": len(reasoned_response)
                }
            }

        except Exception as e:
            self.logger.error(f"Reasoning error: {str(e)}")
            raise

def create_enhanced_rag_system(
    embeddings,
    retriever,
    llm,
    verbose: bool = True
) -> callable:
    """Create an enhanced multi-agent RAG system"""
    
    # Initialize specialized agents
    document_retrieval = DocumentRetrievalAgent(retriever, llm)
    query_rewrite = QueryRewriteAgent(llm)
    reasoning = ReasoningAgent(llm)
    
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    def enhanced_rag_workflow(
        query: str,
        conversation_id: Optional[str] = None
    ) -> QueryResult:
        """Execute enhanced RAG workflow"""
        try:
            # Step 1: Query Rewriting
            query_result = query_rewrite._run(query)
            rewritten_query = query_result["reformulated_query"]
            
            if verbose:
                print(f"🔍 Rewritten Query: {rewritten_query}")

            # Step 2: Document Retrieval
            retrieval_result = document_retrieval._run(rewritten_query)
            retrieved_docs = retrieval_result["documents"]
            context_analysis = retrieval_result["analysis"]
            
            if verbose:
                print(f"📚 Retrieved {len(retrieved_docs)} documents")
                print("Context Analysis Summary:", context_analysis[:200] + "...")

            # Step 3: Reasoning and Synthesis
            reasoning_result = reasoning._run(rewritten_query, context_analysis)
            final_response = reasoning_result["reasoned_response"]
            
            if verbose:
                print("🧠 Generated Response")

            # Create structured result
            result = QueryResult(
                original_query=query,
                rewritten_query=rewritten_query,
                retrieved_documents=retrieved_docs,
                context_analysis=context_analysis,
                reasoned_response=reasoning_result["reasoned_response"],
                final_response=final_response,
                metadata={
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now(),
                    "process_metadata": {
                        "query_rewrite": query_result["metadata"],
                        "retrieval": retrieval_result["metadata"],
                        "reasoning": reasoning_result["metadata"]
                    }
                },
                timestamp=datetime.now()
            )

            # Update memory
            memory.save_context(
                {"input": query},
                {"output": final_response}
            )

            return result

        except Exception as e:
            logging.error(f"RAG workflow error: {str(e)}")
            raise

    return enhanced_rag_workflow

def initialize_enhanced_rag_system(
    persist_directory: str = "./db-mawared",
    model_name: str = "llama-3.1-8b-instant",
    embedding_model: str = "nomic-embed-text"
) -> callable:
    """Initialize the enhanced RAG system with configurable parameters"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load environment variables
        load_dotenv()
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")

        # Initialize embeddings
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            show_progress=False
        )

        # Initialize vector store
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        # Create retriever with configurable parameters
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7
            }
        )

        # Initialize LLM with enhanced configuration
        llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=60,
            max_retries=3
        )

        # Create Enhanced RAG System
        enhanced_rag = create_enhanced_rag_system(
            embeddings,
            retriever,
            llm,
            verbose=True
        )

        logger.info("Enhanced RAG system initialized successfully")
        return enhanced_rag

    except Exception as e:
        logger.error(f"System initialization error: {str(e)}")
        raise

def main():
    """Enhanced main interaction loop"""
    print("\n🤖 مرحباً! أنا نظام متعدد الوكلاء للذكاء الاصطناعي المحسّن.")

    try:
        # Initialize Enhanced RAG system
        rag_system = initialize_enhanced_rag_system()
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        while True:
            user_question = input("\nأدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")

            if user_question.lower() in ['quit', 'خروج']:
                break

            try:
                # Execute enhanced workflow
                with get_openai_callback() as cb:
                    result = rag_system(
                        user_question,
                        conversation_id=conversation_id
                    )

                print("\n📝 الإجابة النهائية:")
                print(result.final_response)

                # Display additional information
                print("\n📊 إحصائيات المعالجة:")
                print(f"- وقت المعالجة: {result.metadata['timestamp']}")
                print(f"- عدد الوثائق المسترجعة: {len(result.retrieved_documents)}")

            except Exception as e:
                print(f"\n❌ خطأ في معالجة السؤال: {str(e)}")

    except Exception as e:
        print(f"\n❌ خطأ في تهيئة النظام: {str(e)}")

    print("\nشكراً لاستخدامك النظام المحسّن متعدد الوكلاء. إلى اللقاء!")

if __name__ == "__main__":
    main()