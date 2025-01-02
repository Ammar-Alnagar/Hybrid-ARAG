import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import normalize
import faiss

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
class TurboQueryResult:
    """Enhanced result structure for Turbo RAG"""
    original_query: str
    transformed_queries: List[str]
    search_results: Dict[str, List[Document]]
    hybrid_results: List[Document]
    reranked_results: List[Document]
    final_response: str
    metadata: Dict[str, Any]
    timestamp: datetime

class QueryTransformer:
    """Advanced query transformation with multiple strategies"""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        self.transform_prompt = PromptTemplate.from_template("""
        Generate diverse query transformations for effective retrieval:
        
        Original Query: {query}
        
        Generate these types of transformations:
        1. Specification: Add specific details and constraints
        2. Generalization: Broaden the scope to related concepts
        3. Perspective Shift: Rephrase from different viewpoints
        4. Temporal Variants: Consider time-based variations
        5. Contextual Expansion: Add domain-specific context
        
        For each type, provide 2 transformed queries.
        Format: type: transformed_query
        """)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def transform_query(self, query: str) -> List[str]:
        """Generate multiple query transformations"""
        try:
            # Get transformed queries
            response = self.llm.invoke(
                self.transform_prompt.format(query=query)
            ).content
            
            # Parse and validate transformations
            transformations = []
            for line in response.split('\n'):
                if ':' in line:
                    _, transformed = line.split(':', 1)
                    transformed = transformed.strip()
                    if transformed and transformed != query:
                        transformations.append(transformed)
            
            # Add original query
            transformations.insert(0, query)
            
            return transformations
            
        except Exception as e:
            self.logger.error(f"Query transformation error: {str(e)}")
            raise

class HybridSearcher:
    """Hybrid search combining dense and sparse retrievers"""
    
    def __init__(self, 
                 embeddings,
                 vector_store: Chroma,
                 sparse_weight: float = 0.3):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.sparse_weight = sparse_weight
        self.dense_weight = 1.0 - sparse_weight
        self.logger = logging.getLogger(__name__)
        
        # Initialize FAISS index for fast similarity search
        self.dimension = 384  # Update based on your embedding dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        
    def _dense_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Perform dense vector search"""
        query_embedding = self.embeddings.embed_documents([query])[0]
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def _sparse_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Perform sparse lexical search"""
        # Implement BM25 or similar sparse retrieval
        # This is a simplified version
        results = []
        documents = self.vector_store.get()["documents"]
        
        # Simple term frequency scoring
        query_terms = set(query.lower().split())
        for doc in documents:
            doc_terms = set(doc.page_content.lower().split())
            score = len(query_terms.intersection(doc_terms)) / len(query_terms)
            results.append((doc, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]
    
    def hybrid_search(self, 
                     query: str,
                     k: int = 10) -> List[Tuple[Document, float]]:
        """Combine dense and sparse search results"""
        try:
            # Get results from both methods
            dense_results = self._dense_search(query, k=k)
            sparse_results = self._sparse_search(query, k=k)
            
            # Combine and normalize scores
            combined_scores = {}
            
            # Process dense results
            for doc, score in dense_results:
                combined_scores[doc] = self.dense_weight * score
            
            # Process sparse results
            for doc, score in sparse_results:
                if doc in combined_scores:
                    combined_scores[doc] += self.sparse_weight * score
                else:
                    combined_scores[doc] = self.sparse_weight * score
            
            # Sort by combined score
            results = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid search error: {str(e)}")
            raise

class ResultReranker:
    """Advanced result reranking with multiple strategies"""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        self.rerank_prompt = PromptTemplate.from_template("""
        Analyze and rank the relevance of retrieved documents:
        
        Query: {query}
        
        Documents:
        {documents}
        
        For each document, assess:
        1. Query Relevance (0-1)
        2. Information Quality (0-1)
        3. Context Alignment (0-1)
        4. Specificity (0-1)
        
        Provide a final combined score (0-1) for each document.
        Format: doc_id: score
        """)
    
    @retry(stop=stop_attempt_number=3, wait_exponential_multiplier=1000)
    def rerank_results(self, 
                      query: str,
                      documents: List[Tuple[Document, float]]) -> List[Document]:
        """Rerank results using LLM-based analysis"""
        try:
            # Format documents for analysis
            docs_text = "\n\n".join([
                f"Doc {i}:\n{doc.page_content[:200]}..."
                for i, (doc, _) in enumerate(documents)
            ])
            
            # Get reranking analysis
            response = self.llm.invoke(
                self.rerank_prompt.format(
                    query=query,
                    documents=docs_text
                )
            ).content
            
            # Parse scores
            scores = {}
            for line in response.split('\n'):
                if ':' in line:
                    doc_id, score = line.split(':')
                    doc_id = int(doc_id.strip().split()[-1])
                    score = float(score.strip())
                    scores[doc_id] = score
            
            # Rerank documents
            reranked = [
                doc for i, (doc, _) in enumerate(documents)
                if i in scores
            ]
            reranked.sort(key=lambda x: scores[documents.index((x, 0))], reverse=True)
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"Reranking error: {str(e)}")
            raise

class TurboRAG:
    """Enhanced RAG system with Turbo optimizations"""
    
    def __init__(self,
                 embeddings,
                 vector_store: Chroma,
                 llm,
                 verbose: bool = True):
        self.query_transformer = QueryTransformer(llm)
        self.hybrid_searcher = HybridSearcher(embeddings, vector_store)
        self.reranker = ResultReranker(llm)
        self.llm = llm
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Response generation prompt
        self.response_prompt = PromptTemplate.from_template("""
        Generate a comprehensive response using the retrieved information:
        
        Query: {query}
        
        Context:
        {context}
        
        Requirements:
        1. Synthesize information from multiple sources
        2. Maintain factual accuracy
        3. Provide specific details and examples
        4. Address all aspects of the query
        5. Use clear and concise language
        
        Generate a well-structured response:
        """)
    
    def process_query(self,
                     query: str,
                     k: int = 10) -> TurboQueryResult:
        """Process query using Turbo RAG pipeline"""
        try:
            # Step 1: Query Transformation
            if self.verbose:
                print("🔄 Transforming query...")
            transformed_queries = self.query_transformer.transform_query(query)
            
            # Step 2: Hybrid Search
            if self.verbose:
                print("🔍 Performing hybrid search...")
            search_results = {}
            for q in transformed_queries:
                results = self.hybrid_searcher.hybrid_search(q, k=k)
                search_results[q] = [doc for doc, _ in results]
            
            # Step 3: Result Aggregation
            all_results = []
            for results in search_results.values():
                all_results.extend(results)
            
            # Remove duplicates while preserving order
            seen = set()
            hybrid_results = []
            for doc in all_results:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    hybrid_results.append(doc)
            
            # Step 4: Reranking
            if self.verbose:
                print("📊 Reranking results...")
            reranked_results = self.reranker.rerank_results(
                query,
                [(doc, 0.0) for doc in hybrid_results[:k]]
            )
            
            # Step 5: Response Generation
            if self.verbose:
                print("✍️ Generating response...")
            context = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(reranked_results[:3])
            ])
            
            response = self.llm.invoke(
                self.response_prompt.format(
                    query=query,
                    context=context
                )
            ).content
            
            # Create result object
            result = TurboQueryResult(
                original_query=query,
                transformed_queries=transformed_queries,
                search_results=search_results,
                hybrid_results=hybrid_results,
                reranked_results=reranked_results,
                final_response=response,
                metadata={
                    "timestamp": datetime.now(),
                    "num_transformations": len(transformed_queries),
                    "num_hybrid_results": len(hybrid_results),
                    "num_reranked": len(reranked_results)
                },
                timestamp=datetime.now()
            )
            
            # Update memory
            self.memory.save_context(
                {"input": query},
                {"output": response}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing error: {str(e)}")
            raise

def initialize_turbo_rag(
    persist_directory: str = "./db-mawared",
    model_name: str = "llama-3.1-8b-instant",
    embedding_model: str = "nomic-embed-text",
    verbose: bool = True
) -> TurboRAG:
    """Initialize Turbo RAG system"""
    
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
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Initialize LLM
        llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=60,
            max_retries=3
        )
        
        # Create Turbo RAG system
        turbo_rag = TurboRAG(
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
            verbose=verbose
        )
        
        logger.info("Turbo RAG system initialized successfully")
        return turbo_rag
    
    except Exception as e:
        logger.error(f"System initialization error: {str(e)}")
        raise

def main():
    """Main interaction loop"""
    print("\n🚀 Welcome to Turbo RAG!")
    
    try:
        # Initialize Turbo RAG system
        rag_system = initialize_turbo_rag()
        
        while True:
            query = input("\nEnter your query (or 'quit' to exit): ")
            
            if query.lower() == 'quit':
                break
            
            try:
                # Process query
                result = rag_system.process_query(query)
                
                print("\n📝 Final Response:")
                print(result.final_response)
                
                if rag_system.verbose:
                    print("\n📊 Processing Statistics:")
                    print(f"- Query Transformations: {len(result.transformed_queries)}")
                    print(f"- Hybrid Results: {len(result.hybrid_results)}")
                    print(f"- Reranked Results: {len(result.reranked_results)}")
                    print(f"- Processing Time: {datetime.now() - result.timestamp}")
                
            except Exception as e:
                print(f"\n❌ Error processing query: {str(e)}")
    
    except Exception as e:
        print(f"\n❌ System initialization error: {str(e)}")
    
    print("\nThank you for using Turbo RAG!")

if __name__ == "__main__":
    main()