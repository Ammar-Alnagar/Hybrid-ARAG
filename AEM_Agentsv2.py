import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
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
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Error handling and logging
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Performance monitoring
import time
import psutil
from prometheus_client import Counter, Histogram, start_http_server

# Add caching
set_llm_cache(InMemoryCache())

# Metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total number of RAG queries processed')
QUERY_DURATION = Histogram('rag_query_duration_seconds', 'Time spent processing RAG queries')
ERROR_COUNTER = Counter('rag_errors_total', 'Total number of errors encountered')

@dataclass
class QueryResult:
    """Enhanced query result container with confidence scores"""
    original_query: str
    rewritten_query: str
    retrieved_documents: List[Document]
    context_analysis: str
    reasoned_response: str
    final_response: str
    confidence_score: float
    metadata: Dict[str, Any]
    timestamp: datetime
    performance_metrics: Dict[str, float]

class DocumentCache:
    """Cache for frequently accessed documents"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def get(self, key: str) -> Optional[Document]:
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None

    def put(self, key: str, doc: Document):
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = doc
        self.access_count[key] = 1

class PerformanceMonitor:
    """Monitor system performance metrics"""
    def __init__(self):
        self.start_time = time.time()

    def get_metrics(self) -> Dict[str, float]:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'process_time': time.time() - self.start_time
        }

class DocumentRetrievalAgent(Tool):
    """Enhanced document retrieval agent with caching and parallel processing"""
    name = "document_retrieval_agent"
    description = "Advanced document retrieval with parallel processing"

    def __init__(self, retriever, llm, chunk_size: int = 1000):
        super().__init__(name=self.name, func=self._run, description=self.description)
        self.retriever = retriever
        self.llm = llm
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        self.cache = DocumentCache()
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def _process_chunk_async(self, chunk: Document) -> Dict[str, Any]:
        """Process document chunk asynchronously"""
        try:
            # Calculate chunk embedding
            text = chunk.page_content
            cache_key = hash(text)
            
            if cached_result := self.cache.get(cache_key):
                return cached_result
            
            # Process chunk
            processed_chunk = {
                'content': text,
                'embedding': await self.llm.aembed_query(text),
                'metadata': chunk.metadata
            }
            
            self.cache.put(cache_key, processed_chunk)
            return processed_chunk
            
        except Exception as e:
            self.logger.error(f"Chunk processing error: {str(e)}")
            raise

    async def _parallel_process_documents(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Process documents in parallel"""
        tasks = [self._process_chunk_async(doc) for doc in docs]
        return await asyncio.gather(*tasks)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Asynchronous document retrieval with parallel processing"""
        try:
            # Retrieve documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Process documents in parallel
            processed_docs = await self._parallel_process_documents(docs)
            
            # Enhanced analysis prompt
            analysis_prompt = PromptTemplate.from_template("""
            Perform advanced document analysis:
            
            Query: {query}
            Documents: {documents}
            
            Analysis Requirements:
            1. Semantic Relevance:
               - Topic alignment
               - Information density
               - Context coverage
            
            2. Information Quality:
               - Source credibility
               - Data freshness
               - Completeness
            
            3. Cross-Document Analysis:
               - Information overlap
               - Contradictions
               - Complementary insights
            
            4. Query-Specific Value:
               - Direct relevance score
               - Indirect utility
               - Knowledge gaps
            
            Provide comprehensive analysis addressing all points.
            """)
            
            # Generate analysis with processed documents
            analysis_input = analysis_prompt.format(
                query=query,
                documents=json.dumps(processed_docs, indent=2)
            )
            
            context_analysis = await self.llm.ainvoke(analysis_input)
            
            return {
                "documents": docs,
                "processed_docs": processed_docs,
                "analysis": context_analysis.content,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "doc_count": len(docs),
                    "processing_time": time.time()
                }
            }

        except Exception as e:
            ERROR_COUNTER.inc()
            self.logger.error(f"Async document retrieval error: {str(e)}")
            raise

class ConfidenceEstimator:
    """Estimate confidence scores for responses"""
    def __init__(self):
        self.metrics = {
            'retrieval_score': 0.3,
            'reasoning_score': 0.4,
            'context_score': 0.3
        }

    def calculate_confidence(
        self,
        retrieval_quality: float,
        reasoning_quality: float,
        context_quality: float
    ) -> float:
        """Calculate weighted confidence score"""
        return (
            self.metrics['retrieval_score'] * retrieval_quality +
            self.metrics['reasoning_score'] * reasoning_quality +
            self.metrics['context_score'] * context_quality
        )

class ResponseValidator:
    """Validate and improve response quality"""
    def __init__(self, llm):
        self.llm = llm
        self.validation_prompt = PromptTemplate.from_template("""
        Validate and improve the following response:
        
        Response: {response}
        
        Validation Criteria:
        1. Factual Accuracy
        2. Logical Coherence
        3. Completeness
        4. Clarity
        5. Bias Detection
        
        Provide:
        1. Validation results
        2. Suggested improvements
        3. Confidence score (0-1)
        """)

    async def validate(self, response: str) -> Tuple[str, float]:
        """Validate response and return improved version with confidence"""
        validation_input = self.validation_prompt.format(response=response)
        validation_result = await self.llm.ainvoke(validation_input)
        
        # Parse validation result
        result_lines = validation_result.content.split('\n')
        improved_response = '\n'.join(result_lines[:-1])
        confidence = float(result_lines[-1])
        
        return improved_response, confidence

def create_enhanced_rag_system(
    embeddings,
    retriever,
    llm,
    verbose: bool = True
) -> callable:
    """Create enhanced RAG system with new features"""
    
    # Initialize components
    document_retrieval = DocumentRetrievalAgent(retriever, llm)
    query_rewrite = QueryRewriteAgent(llm)
    reasoning = ReasoningAgent(llm)
    confidence_estimator = ConfidenceEstimator()
    response_validator = ResponseValidator(llm)
    performance_monitor = PerformanceMonitor()
    
    # Initialize memory with metadata
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
        input_key="input"
    )

    async def enhanced_rag_workflow(
        query: str,
        conversation_id: Optional[str] = None
    ) -> QueryResult:
        """Execute enhanced async RAG workflow"""
        try:
            QUERY_COUNTER.inc()
            start_time = time.time()
            
            # Step 1: Query Rewriting
            query_result = await query_rewrite._arun(query)
            rewritten_query = query_result["reformulated_query"]
            
            if verbose:
                print(f"🔍 Rewritten Query: {rewritten_query}")

            # Step 2: Parallel Document Retrieval
            retrieval_result = await document_retrieval._arun(rewritten_query)
            retrieved_docs = retrieval_result["documents"]
            context_analysis = retrieval_result["analysis"]
            
            if verbose:
                print(f"📚 Retrieved {len(retrieved_docs)} documents")

            # Step 3: Enhanced Reasoning
            reasoning_result = await reasoning._arun(rewritten_query, context_analysis)
            initial_response = reasoning_result["reasoned_response"]
            
            # Step 4: Response Validation and Improvement
            final_response, confidence = await response_validator.validate(initial_response)
            
            # Calculate performance metrics
            performance_metrics = performance_monitor.get_metrics()
            
            # Create comprehensive result
            result = QueryResult(
                original_query=query,
                rewritten_query=rewritten_query,
                retrieved_documents=retrieved_docs,
                context_analysis=context_analysis,
                reasoned_response=initial_response,
                final_response=final_response,
                confidence_score=confidence,
                metadata={
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now(),
                    "process_metadata": {
                        "query_rewrite": query_result["metadata"],
                        "retrieval": retrieval_result["metadata"],
                        "reasoning": reasoning_result["metadata"]
                    }
                },
                timestamp=datetime.now(),
                performance_metrics=performance_metrics
            )

            # Update memory with metadata
            memory.save_context(
                {"input": query},
                {
                    "output": final_response,
                    "metadata": result.metadata
                }
            )

            # Record query duration
            QUERY_DURATION.observe(time.time() - start_time)

            return result

        except Exception as e:
            ERROR_COUNTER.inc()
            logging.error(f"Enhanced RAG workflow error: {str(e)}")
            raise

    return enhanced_rag_workflow

async def main():
    """Enhanced async main interaction loop with metrics"""
    print("\n🤖 مرحباً! أنا نظام متعدد الوكلاء للذكاء الاصطناعي المتقدم.")

    # Start metrics server
    start_http_server(8000)

    try:
        # Initialize system
        rag_system = initialize_enhanced_rag_system()
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        while True:
            user_question = input("\nأدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")

            if user_question.lower() in ['quit', 'خروج']:
                break

            try:
                # Execute enhanced async workflow
                with get_openai_callback() as cb:
                    result = await rag_system(
                        user_question,
                        conversation_id=conversation_id
                    )

                print("\n📝 الإجابة النهائية:")
                print(result.final_response)

                # Display enhanced metrics
                print("\n📊 معلومات المعالجة:")
                print(f"- درجة الثقة: {result.confidence_score:.2%}")
                print(f"- وقت المعالجة: {result.performance_metrics['process_time']:.2f}s")
                print(f"- استخدام المعالج: {result.performance_metrics['cpu_percent']}%")
                print(f"- استخدام الذاكرة: {result.performance_metrics['memory_percent']}%")

            except Exception as e:
                print(f"\n❌ خطأ في معالجة السؤال: {str(e)}")

    except Exception as e:
        print(f"\n❌ خطأ في تهيئة النظام: {str(e)}")

    print("\nشكراً لاستخدامك النظام المتقدم متعدد الوكلاء. إلى اللقاء!")

if __name__ == "__main__":
    asyncio.run(main())