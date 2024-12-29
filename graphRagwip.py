import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
from collections import defaultdict
import spacy
from pathlib import Path
import json

# Graph Database
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship

# LangChain and Core Imports
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import Document

# Visualization
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

@dataclass
class GraphQueryResult:
    """Enhanced query result with graph information"""
    query: str
    subgraph: nx.Graph
    paths: List[List[str]]
    nodes: Set[str]
    relationships: List[tuple]
    context: str
    answer: str
    confidence: float
    metadata: Dict[str, Any]

class KnowledgeGraphBuilder:
    """Build and maintain knowledge graph from documents"""
    
    def __init__(self):
        # Initialize NLP
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize graph
        self.graph = nx.Graph()
        # Node and relationship tracking
        self.nodes = set()
        self.relationships = defaultdict(set)
        
    def extract_entities(self, text: str) -> List[tuple]:
        """Extract entities and relationships from text"""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
            
        # Extract subject-verb-object patterns
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ == "dobj":
                        obj = child.text
                        entities.append((subject, verb, obj))
        
        return entities

    def add_to_graph(self, entities: List[tuple]):
        """Add extracted entities to knowledge graph"""
        for entity in entities:
            if len(entity) == 2:  # Named entity
                node_text, node_type = entity
                self.graph.add_node(
                    node_text, 
                    type=node_type,
                    weight=1.0
                )
                self.nodes.add(node_text)
                
            elif len(entity) == 3:  # Relationship
                subj, pred, obj = entity
                # Add nodes
                self.graph.add_node(subj, type="Entity")
                self.graph.add_node(obj, type="Entity")
                # Add edge
                self.graph.add_edge(
                    subj, 
                    obj, 
                    relationship=pred,
                    weight=1.0
                )
                self.nodes.add(subj)
                self.nodes.add(obj)
                self.relationships[pred].add((subj, obj))

    def process_document(self, doc: Document):
        """Process document and update knowledge graph"""
        # Extract entities and relationships
        extracted = self.extract_entities(doc.page_content)
        # Add to graph
        self.add_to_graph(extracted)
        # Update node weights based on document metadata
        relevance = doc.metadata.get('relevance_score', 1.0)
        for node in self.graph.nodes():
            self.graph.nodes[node]['weight'] *= relevance

    def get_subgraph(self, nodes: Set[str], depth: int = 2) -> nx.Graph:
        """Extract subgraph around specified nodes"""
        subgraph_nodes = set(nodes)
        # Add neighboring nodes up to specified depth
        for _ in range(depth):
            neighbors = set()
            for node in subgraph_nodes:
                neighbors.update(self.graph.neighbors(node))
            subgraph_nodes.update(neighbors)
        
        return self.graph.subgraph(subgraph_nodes)

    def find_paths(self, start_node: str, end_node: str, max_depth: int = 3) -> List[List[str]]:
        """Find paths between nodes in graph"""
        try:
            paths = nx.all_simple_paths(
                self.graph, 
                start_node, 
                end_node, 
                cutoff=max_depth
            )
            return list(paths)
        except nx.NetworkXNoPath:
            return []

    def visualize_graph(self, output_path: str = "knowledge_graph.html"):
        """Create interactive visualization of knowledge graph"""
        net = Network(
            height="750px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="black"
        )
        
        # Add nodes
        for node in self.graph.nodes(data=True):
            node_id, data = node
            net.add_node(
                node_id,
                title=f"Type: {data.get('type', 'Unknown')}",
                size=data.get('weight', 1.0) * 10
            )
            
        # Add edges
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            net.add_edge(
                source,
                target,
                title=data.get('relationship', 'related'),
                weight=data.get('weight', 1.0)
            )
            
        # Save visualization
        net.show(output_path)

class GraphRAGSystem:
    """Graph-enhanced RAG system"""
    
    def __init__(
        self,
        embeddings,
        retriever,
        llm,
        graph_builder: KnowledgeGraphBuilder
    ):
        self.embeddings = embeddings
        self.retriever = retriever
        self.llm = llm
        self.graph_builder = graph_builder
        self.neo4j_client = None  # Optional Neo4j connection
        
    def connect_neo4j(self, uri: str, user: str, password: str):
        """Connect to Neo4j database"""
        self.neo4j_client = GraphDatabase.driver(uri, auth=(user, password))

    def process_documents(self, documents: List[Document]):
        """Process documents and build knowledge graph"""
        for doc in documents:
            self.graph_builder.process_document(doc)

    def extract_query_entities(self, query: str) -> Set[str]:
        """Extract relevant entities from query"""
        entities = self.graph_builder.extract_entities(query)
        return {ent[0] for ent in entities if len(ent) == 2}

    def get_graph_context(
        self,
        query: str,
        retrieved_docs: List[Document]
    ) -> Dict[str, Any]:
        """Get graph-based context for query"""
        # Extract query entities
        query_entities = self.extract_query_entities(query)
        
        # Get relevant subgraph
        subgraph = self.graph_builder.get_subgraph(query_entities)
        
        # Find paths between entities
        paths = []
        entities = list(query_entities)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                paths.extend(
                    self.graph_builder.find_paths(
                        entities[i],
                        entities[j]
                    )
                )
        
        # Get relationships
        relationships = []
        for edge in subgraph.edges(data=True):
            source, target, data = edge
            relationships.append(
                (source, data.get('relationship', 'related'), target)
            )
        
        return {
            "subgraph": subgraph,
            "paths": paths,
            "relationships": relationships
        }

    async def query(
        self,
        query: str,
        conversation_id: Optional[str] = None
    ) -> GraphQueryResult:
        """Execute graph-enhanced query"""
        try:
            # Retrieve documents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            # Process documents into graph
            self.process_documents(retrieved_docs)
            
            # Get graph context
            graph_context = self.get_graph_context(query, retrieved_docs)
            
            # Create enhanced prompt with graph information
            prompt = PromptTemplate.from_template("""
            Answer the following query using both document and graph context:
            
            Query: {query}
            
            Document Context:
            {doc_context}
            
            Graph Context:
            - Related Entities: {entities}
            - Relationships: {relationships}
            - Discovered Paths: {paths}
            
            Provide a comprehensive answer that:
            1. Integrates information from both documents and graph
            2. Explains relationships between relevant entities
            3. Highlights any indirect connections found through graph paths
            
            Answer:
            """)
            
            # Format document context
            doc_context = "\n".join(
                doc.page_content for doc in retrieved_docs
            )
            
            # Format graph context
            prompt_input = {
                "query": query,
                "doc_context": doc_context,
                "entities": list(graph_context["subgraph"].nodes()),
                "relationships": graph_context["relationships"],
                "paths": graph_context["paths"]
            }
            
            # Generate answer
            response = await self.llm.ainvoke(
                prompt.format(**prompt_input)
            )
            
            # Calculate confidence based on graph coverage
            query_entities = self.extract_query_entities(query)
            coverage = len(
                query_entities.intersection(
                    graph_context["subgraph"].nodes()
                )
            ) / len(query_entities)
            confidence = (coverage + 1) / 2  # Combine with response confidence
            
            # Create result
            result = GraphQueryResult(
                query=query,
                subgraph=graph_context["subgraph"],
                paths=graph_context["paths"],
                nodes=set(graph_context["subgraph"].nodes()),
                relationships=graph_context["relationships"],
                context=doc_context,
                answer=response.content,
                confidence=confidence,
                metadata={
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now(),
                    "doc_count": len(retrieved_docs),
                    "entity_count": len(graph_context["subgraph"].nodes()),
                    "relationship_count": len(graph_context["relationships"])
                }
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Graph RAG error: {str(e)}")
            raise

def initialize_graph_rag():
    """Initialize Graph RAG system"""
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        show_progress=False
    )
    
    # Initialize vector store
    db = Chroma(
        persist_directory="./db-graph-rag",
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
        timeout=60
    )
    
    # Initialize graph builder
    graph_builder = KnowledgeGraphBuilder()
    
    # Create Graph RAG system
    graph_rag = GraphRAGSystem(
        embeddings,
        retriever,
        llm,
        graph_builder
    )
    
    return graph_rag

async def main():
    """Interactive Graph RAG interface"""
    print("\n🤖 مرحباً! أنا نظام RAG المعزز بالرسم البياني.")
    
    try:
        # Initialize system
        graph_rag = initialize_graph_rag()
        conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while True:
            query = input("\nأدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")
            
            if query.lower() in ['quit', 'خروج']:
                break
                
            try:
                # Execute query
                result = await graph_rag.query(
                    query,
                    conversation_id=conversation_id
                )
                
                print("\n📝 الإجابة:")
                print(result.answer)
                
                print("\n📊 معلومات الرسم البياني:")
                print(f"- عدد العقد: {len(result.nodes)}")
                print(f"- عدد العلاقات: {len(result.relationships)}")
                print(f"- المسارات المكتشفة: {len(result.paths)}")
                print(f"- درجة الثقة: {result.confidence:.2%}")
                
                # Visualize subgraph
                graph_rag.graph_builder.visualize_graph(
                    f"query_graph_{conversation_id}.html"
                )
                print("\n🔍 تم إنشاء تصور الرسم البياني!")
                
            except Exception as e:
                print(f"\n❌ خطأ في معالجة السؤال: {str(e)}")
                
    except Exception as e:
        print(f"\n❌ خطأ في تهيئة النظام: {str(e)}")
        
    print("\nشكراً لاستخدامك نظام RAG المعزز بالرسم البياني. إلى اللقاء!")

if __name__ == "__main__":
    asyncio.run(main())