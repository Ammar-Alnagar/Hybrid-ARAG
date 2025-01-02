import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from networkx import DiGraph, shortest_path
import networkx as nx
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
class KnowledgeNode:
    """Representation of a node in the knowledge graph"""
    id: str
    content: str
    node_type: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None

@dataclass
class KnowledgeEdge:
    """Representation of an edge in the knowledge graph"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    metadata: Dict[str, Any]

@dataclass
class GraphQueryResult:
    """Enhanced query result with graph-based analysis"""
    original_query: str
    rewritten_query: str
    retrieved_nodes: List[KnowledgeNode]
    graph_context: str
    path_analysis: str
    reasoning_chains: List[List[str]]
    final_response: str
    metadata: Dict[str, Any]
    timestamp: datetime

class KnowledgeGraphManager:
    """Manager for the knowledge graph operations"""
    def __init__(self, embeddings):
        self.graph = DiGraph()
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the knowledge graph"""
        if not node.embeddings:
            node.embeddings = self.embeddings.embed_documents([node.content])[0]
        
        self.graph.add_node(
            node.id,
            content=node.content,
            node_type=node.node_type,
            metadata=node.metadata,
            embeddings=node.embeddings
        )

    def add_edge(self, edge: KnowledgeEdge) -> None:
        """Add an edge to the knowledge graph"""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation_type=edge.relation_type,
            weight=edge.weight,
            metadata=edge.metadata
        )

    def find_relevant_subgraph(self, query: str, max_nodes: int = 5) -> DiGraph:
        """Find relevant subgraph based on query embeddings"""
        query_embedding = self.embeddings.embed_documents([query])[0]
        
        # Calculate similarity scores for all nodes
        similarities = []
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            similarity = self._cosine_similarity(
                query_embedding,
                node_data['embeddings']
            )
            similarities.append((node_id, similarity))
        
        # Sort and select top nodes
        top_nodes = sorted(similarities, key=lambda x: x[1], reverse=True)[:max_nodes]
        relevant_nodes = {node_id for node_id, _ in top_nodes}
        
        # Extract subgraph with intermediate nodes
        subgraph = self._extract_connected_subgraph(relevant_nodes)
        return subgraph

    def _extract_connected_subgraph(self, seed_nodes: Set[str]) -> DiGraph:
        """Extract a connected subgraph containing seed nodes"""
        subgraph = DiGraph()
        
        # Add all paths between seed nodes
        for source in seed_nodes:
            for target in seed_nodes:
                if source != target:
                    try:
                        path = shortest_path(self.graph, source, target)
                        for i in range(len(path) - 1):
                            current = path[i]
                            next_node = path[i + 1]
                            if not subgraph.has_node(current):
                                subgraph.add_node(
                                    current,
                                    **self.graph.nodes[current]
                                )
                            if not subgraph.has_node(next_node):
                                subgraph.add_node(
                                    next_node,
                                    **self.graph.nodes[next_node]
                                )
                            subgraph.add_edge(
                                current,
                                next_node,
                                **self.graph.edges[current, next_node]
                            )
                    except nx.NetworkXNoPath:
                        continue
        
        return subgraph

    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        return dot_product / (norm1 * norm2)

class GraphTraversalAgent(Tool):
    """Agent for graph-based knowledge traversal and analysis"""
    name = "graph_traversal_agent"
    description = "Graph-based knowledge traversal and analysis"

    def __init__(self, graph_manager: KnowledgeGraphManager, llm):
        super().__init__(name=self.name, func=self._run, description=self.description)
        self.graph_manager = graph_manager
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _run(self, query: str) -> Dict[str, Any]:
        """Execute graph-based knowledge traversal"""
        try:
            # Find relevant subgraph
            subgraph = self.graph_manager.find_relevant_subgraph(query)
            
            # Analyze paths and relationships
            analysis_prompt = PromptTemplate.from_template("""
            Analyze the knowledge graph context:
            
            Query: {query}
            
            Graph Context:
            {graph_context}
            
            Required Analysis:
            1. Path Analysis:
               - Key knowledge paths
               - Connection strengths
               - Information flow
            
            2. Relationship Analysis:
               - Direct connections
               - Indirect influences
               - Knowledge gaps
            
            3. Context Integration:
               - Central concepts
               - Supporting evidence
               - Conflicting information
            
            Please provide a structured analysis of the knowledge graph.
            """)

            # Format graph context
            graph_context = self._format_graph_context(subgraph)
            
            # Generate analysis
            analysis_input = analysis_prompt.format(
                query=query,
                graph_context=graph_context
            )
            
            graph_analysis = self.llm.invoke(analysis_input).content
            
            return {
                "subgraph": subgraph,
                "analysis": graph_analysis,
                "metadata": {
                    "node_count": subgraph.number_of_nodes(),
                    "edge_count": subgraph.number_of_edges(),
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Graph traversal error: {str(e)}")
            raise

    def _format_graph_context(self, graph: DiGraph) -> str:
        """Format graph context for analysis"""
        context_parts = []
        
        # Add nodes
        context_parts.append("Nodes:")
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            context_parts.append(
                f"- {node_id} ({node_data['node_type']}): {node_data['content'][:100]}..."
            )
        
        # Add edges
        context_parts.append("\nRelationships:")
        for source, target, data in graph.edges(data=True):
            context_parts.append(
                f"- {source} --[{data['relation_type']}]--> {target}"
            )
        
        return "\n".join(context_parts)

def create_enhanced_cag_system(
    embeddings,
    graph_manager: KnowledgeGraphManager,
    llm,
    verbose: bool = True
) -> callable:
    """Create an enhanced CAG system"""
    
    # Initialize specialized agents
    graph_traversal = GraphTraversalAgent(graph_manager, llm)
    query_rewrite = QueryRewriteAgent(llm)
    reasoning = ReasoningAgent(llm)
    
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    def enhanced_cag_workflow(
        query: str,
        conversation_id: Optional[str] = None
    ) -> GraphQueryResult:
        """Execute enhanced CAG workflow"""
        try:
            # Step 1: Query Rewriting
            query_result = query_rewrite._run(query)
            rewritten_query = query_result["reformulated_query"]
            
            if verbose:
                print(f"🔍 Rewritten Query: {rewritten_query}")
            
            # Step 2: Graph Traversal
            traversal_result = graph_traversal._run(rewritten_query)
            subgraph = traversal_result["subgraph"]
            graph_analysis = traversal_result["analysis"]
            
            if verbose:
                print(f"📊 Analyzed subgraph with {subgraph.number_of_nodes()} nodes")
                print("Graph Analysis Summary:", graph_analysis[:200] + "...")
            
            # Step 3: Reasoning with Graph Context
            reasoning_result = reasoning._run(rewritten_query, graph_analysis)
            final_response = reasoning_result["reasoned_response"]
            
            if verbose:
                print("🧠 Generated Response")
            
            # Create structured result
            result = GraphQueryResult(
                original_query=query,
                rewritten_query=rewritten_query,
                retrieved_nodes=[
                    KnowledgeNode(
                        id=node_id,
                        content=data['content'],
                        node_type=data['node_type'],
                        metadata=data['metadata']
                    )
                    for node_id, data in subgraph.nodes(data=True)
                ],
                graph_context=graph_analysis,
                path_analysis=traversal_result["analysis"],
                reasoning_chains=self._extract_reasoning_chains(subgraph),
                final_response=final_response,
                metadata={
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now(),
                    "process_metadata": {
                        "query_rewrite": query_result["metadata"],
                        "traversal": traversal_result["metadata"],
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
            logging.error(f"CAG workflow error: {str(e)}")
            raise
    
    def _extract_reasoning_chains(graph: DiGraph) -> List[List[str]]:
        """Extract reasoning chains from the graph"""
        chains = []
        # Find all simple paths between nodes
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(graph, source, target))
                        chains.extend([
                            [graph.nodes[node]['content'] for node in path]
                            for path in paths
                        ])
                    except nx.NetworkXNoPath:
                        continue
        return chains
    
    return enhanced_cag_workflow

def initialize_enhanced_cag_system(
    persist_directory: str = "./db-mawared",
    model_name: str = "llama-3.1-8b-instant",
    embedding_model: str = "nomic-embed-text"
) -> callable:
    """Initialize the enhanced CAG system"""
    
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
        
        # Initialize knowledge graph manager
        graph_manager = KnowledgeGraphManager(embeddings)
        
        # Initialize LLM
        llm = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=60,
            max_retries=3
        )
        
        # Create Enhanced CAG System
        enhanced_cag = create_enhanced_cag_system(
            embeddings,
            graph_manager,
            llm,
            verbose=True
        )
        
        logger.info("Enhanced CAG system initialized successfully")
        return enhanced_cag
    
    except Exception as e:
        logger.error(f"System initialization error: {str(e)}")
        raise

def main():
    """Enhanced main interaction loop"""
    print("\n🤖 مرحباً! أنا نظام CAG المحسّن للذكاء الاصطناعي.")
    
    try:
        # Initialize Enhanced CAG system
        cag_system = initialize_enhanced_cag_system()
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while True:
            user_question = input("\nأدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")
            
            if user_question.lower() in ['quit', 'خروج']:
                break
            
            try:
                # Execute enhanced workflow
                with get_openai_callback() as cb:
                    result = cag_system(
                        user_question,
                        conversation_id=conversation_id
                    )
                
                print("\n📝 الإجابة النهائية:")
                print(result.final_response)
                
                # Display additional information
                print("\n📊 إحصائيات المعالجة:")
                print(f"- وقت المعالجة: {result.metadata['timestamp']}")
                print(f"- عدد العقد المسترجعة: {len(result.retrieved_nodes)}")
                print(f"- عدد سلاسل الاستدلال: {len(result.reasoning_chains)}")
                
            except Exception as e:
                print(f"\n❌ خطأ في معالجة السؤال: {str(e)}")