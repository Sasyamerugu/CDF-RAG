import os
from typing import TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, END
from config import CONFIG
from dotenv import load_dotenv

load_dotenv()

class QueryState(TypedDict):
    query: str

class QueryRefiner:
    def __init__(self):
        self.model = CONFIG.get("QUERY_REFINEMENT_MODEL", "gpt-4")
        self.client = OpenAI(
            api_key=CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
    
    def refine(self, query: str) -> str:
        prompt = f"Refine the following query for better retrieval:\n{query}"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Refine complex queries for enhanced information retrieval."},
                    {"role": "user", "content": prompt}
                ]
            )
            # Safely extract content from the response, handling None and different response shapes
            choice = None
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
            elif isinstance(response, dict) and response.get("choices"):
                choice = response["choices"][0]
            if not choice:
                return query
            # message may be an attribute or a dict; content may be in different fields
            msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
            content = None
            if msg:
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            else:
                content = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None) or getattr(choice, "content", None) or (choice.get("content") if isinstance(choice, dict) else None)
            if not content:
                return query
            return content.strip()
        except Exception as e:
            print(f"⚠️ Refinement Error: {e}")
            return query

class QueryRefinementAgent:
    def __init__(self):
        self.refiner = QueryRefiner()
    
    def optimize_query(self, state: dict) -> dict:
        return {"query": self.refiner.refine(state["query"])}