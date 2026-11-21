import os
from openai import OpenAI
from config import CONFIG
from dotenv import load_dotenv
from agents.llm_inference_response_generator import LLMResponseGenerator

load_dotenv()

class HallucinationDetector:
    def __init__(self):
        self.model = CONFIG.get("HALLUCINATION_DETECTION_MODEL", "gpt-4")
        # Added base_url for Gemini compatibility
        self.client = OpenAI(
            api_key=CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    def detect(self, response: str, structured_explanation: str) -> bool:
        """
        Returns True if hallucination is detected (The response is NOT aligned).
        """
        prompt = f"""
You are a hallucination detection model. Determine whether the following response accurately reflects the given structured explanation.

Structured Explanation:
{structured_explanation}

Generated Response:
{response}

Answer with only "Yes" or "No":
- Yes → response is factually aligned
- No → response includes hallucinated content
"""

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in hallucination detection."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = (result.choices[0].message.content or "").lower().strip()
        
        # CORRECTED LOGIC: 
        # If the answer is "no" (meaning NOT aligned), return True (Hallucination detected)
        return "no" in answer

class HallucinationCorrectionAgent:

    def __init__(self):
        self.detector = HallucinationDetector()
        self.generator = LLMResponseGenerator()

    def correct_hallucinations(self, state):

        hallucinated = self.detector.detect(
            response=state["response"],
            structured_explanation=state.get("rewritten_knowledge", "")
        )

        response = state["response"]

        if hallucinated:
            response = self.generator.generate(
                knowledge=state.get("rewritten_knowledge", ""),
                original_query=state["query"],
                force_retrieval=True
            )

        return {
            "is_hallucination": hallucinated,
            "response": response
        }