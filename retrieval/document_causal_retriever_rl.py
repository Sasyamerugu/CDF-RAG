import pickle
from pinecone import Pinecone
import networkx as nx
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from typing import TypedDict
from config import CONFIG

class RetrievalState(TypedDict):
    query: str
    retrieved_docs: list[str]
    causal_docs: list[str]

class DocumentRetriever:
    def __init__(self):
        # Modern Pinecone V3 initialization
        self.pc = Pinecone(api_key=CONFIG["PINECONE_API_KEY"])
        # Pointing explicitly to your verified index
        self.index_name = CONFIG.get("PINECONE_INDEX", "cdf-rag-index")
        self.index = self.pc.Index(self.index_name)
        
        # Local 384-dimension embedding model
        self.encoder = SentenceTransformer(CONFIG["ENCODER_MODEL"])
    
    def retrieve(self, query: str, top_k: int = 5):
        try:
            query_vector = self.encoder.encode(query).tolist()
            results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            
            docs = []
            matches = getattr(results, 'matches', [])
            
            for match in matches:
                meta = getattr(match, 'metadata', {})
                
                # 100% Pylance-safe extraction (No dot notation)
                if isinstance(meta, dict):
                    docs.append(meta.get("text", str(meta)))
                else:
                    docs.append(getattr(meta, "text", str(meta)))
                    
            return docs
        except Exception as e:
            print(f"⚠️ Pinecone Retrieval Error: {e}")
            return []

class CausalGraphRetriever:
    def __init__(self):
        # Safely loading the graph bypassing the deprecated nx.read_gpickle
        with open(CONFIG["CAUSAL_GRAPH_PATH"], "rb") as f:
            self.graph = pickle.load(f)
    
    def retrieve(self, query: str):
        related_nodes = []
        for node in self.graph.nodes:
            if query.lower() in str(node).lower():
                related_nodes.append(node)
                related_nodes.extend(nx.descendants(self.graph, node))
        return list(set(related_nodes))

class AdaptiveRetrieverAgent:
    def __init__(self):
        self.doc_retriever = DocumentRetriever()
        self.causal_retriever = CausalGraphRetriever()
    
    def retrieve_knowledge(self, state: RetrievalState) -> RetrievalState:
        query = state['query']
        state['retrieved_docs'] = self.doc_retriever.retrieve(query)
        state['causal_docs'] = self.causal_retriever.retrieve(query)
        return state

def build_retrieval_graph():
    builder = StateGraph(RetrievalState)
    builder.add_node("retrieve_knowledge", AdaptiveRetrieverAgent().retrieve_knowledge)
    builder.set_entry_point("retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", END)
    return builder.compile()