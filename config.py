import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# INJECT GEMINI COMPATIBILITY: 
# This tells the existing OpenAI library to route requests to Google instead
os.environ["OPENAI_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai/"

CONFIG = {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "dhanush_password",  # Updated to match the Docker container you started

    # Feed the Gemini key into the OpenAI variable
    "OPENAI_API_KEY": os.getenv("GOOGLE_API_KEY"),  
    
    # Swap out GPT-4 for the Gemini 1.5 Flash model
    "KNOWLEDGE_REWRITING_MODEL": "gemini-flash-latest",
    "QUERY_REFINEMENT_MODEL": "gemini-flash-latest",
    "LLM_MODEL": "gemini-flash-latest",
    "HALLUCINATION_DETECTION_MODEL": "gemini-flash-latest",

    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),

    # Keep the default local Sentence Transformer
    "ENCODER_MODEL": "BAAI/bge-base-en-v1.5",  
    "CAUSAL_GRAPH_PATH": "causal_graph.pkl",
}