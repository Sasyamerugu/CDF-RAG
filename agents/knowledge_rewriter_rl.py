import os
from typing import TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph, END
from config import CONFIG
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ FIX 1: Define schema using TypedDict for modern LangGraph compatibility
class RewritingState(TypedDict):
    query: str
    retrieved_docs: list[str]
    causal_docs: list[str]
    rewritten_knowledge: str

# ✅ LLM wrapper for knowledge structuring
class KnowledgeRewriter:
    def __init__(self):
        self.model = CONFIG.get("KNOWLEDGE_REWRITING_MODEL", "gpt-4")
        
        # ✅ FIX 2: Added base_url for Gemini compatibility
        self.client = OpenAI(
            api_key=CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    def rewrite(self, documents: list[str], query: str) -> str:
        context = "\n".join(documents) if documents else "No external context found."
        prompt = f"""
You are a medical reasoning assistant. Your task is to synthesize the following evidence and rewrite it into a coherent explanation that answers the query:

Query: {query}

Evidence:
{context}

Instructions:
- Group related causes together.
- Clarify any multi-hop cause-effect chains.
- Use medically accurate language.
- Return a well-structured explanation (2–3 paragraphs).
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful medical explanation assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"⚠️ Knowledge Rewriter Error: {e}")
            return context # Fallback to raw text if rewriting fails

# ✅ Agent class for integration into LangGraph
class KnowledgeRewritingAgent:
    def __init__(self):
        self.rewriter = KnowledgeRewriter()

    def optimize_rewriting(self, state: RewritingState) -> dict:
        combined_docs = state.get('retrieved_docs', []) + state.get('causal_docs', [])
        structured_summary = self.rewriter.rewrite(combined_docs, state['query'])
        # Return only the updated key for LangGraph state injection
        return {"rewritten_knowledge": structured_summary}

# ✅ FIX 3: Modern LangGraph flow definition using END
builder = StateGraph(RewritingState)
builder.add_node("optimize_rewriting", KnowledgeRewritingAgent().optimize_rewriting)
builder.set_entry_point("optimize_rewriting")
builder.add_edge("optimize_rewriting", END)
graph = builder.compile()

# ✅ Debug runner (for standalone testing)
if __name__ == "__main__":
    test_state = {
        "query": "What causes heart disease?",
        "retrieved_docs": [
            "Hypercholesterolemia causes heart disease.",
            "Obstructive sleep apnea contributes to cardiovascular conditions."
        ],
        "causal_docs": [
            "Hypertension is a common intermediary between stress and heart disease.",
            "Smoking leads to atherosclerosis, which leads to heart disease."
        ],
        "rewritten_knowledge": ""
    }
    final_state = graph.invoke(test_state)
    print("\n🔹 Final Structured Explanation:\n")
    print(final_state.get("rewritten_knowledge"))