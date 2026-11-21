import os
from openai import OpenAI
from pydantic import BaseModel
from config import CONFIG
from dotenv import load_dotenv

load_dotenv()

# ✅ Set up OpenAI client with custom routing support
class LLMResponseGenerator:
    def __init__(self):
        self.model = CONFIG.get("LLM_MODEL", "gpt-4")
        # FIX: Explicitly include base_url to route requests to your free Gemini endpoint
        self.client = OpenAI(
            api_key=CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    def generate(self, knowledge: str, original_query: str, force_retrieval: bool = False) -> str:
        prompt = f"""
    You are a helpful and domain-agnostic assistant. Read the following structured explanation and generate a final answer to the original query.

    Query: {original_query}

    Structured Knowledge:
    {knowledge}

    Instructions:
    - Do NOT mention you're limited to medical topics
    - Provide a confident, natural answer for ANY domain
    - Use a clear, user-friendly tone in 1–2 paragraphs
    """

        if force_retrieval:
            prompt = "Revised Retrieval Required: " + prompt

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medically accurate and user-friendly assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""




