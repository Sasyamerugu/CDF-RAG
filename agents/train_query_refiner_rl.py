import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from openai import OpenAI
from config import CONFIG
from dotenv import load_dotenv

load_dotenv()

# ✅ Initialize local 384-dimension embedding model
print("🔄 Loading local SentenceTransformer...")
encoder = SentenceTransformer(CONFIG.get("ENCODER_MODEL", "all-MiniLM-L6-v2"))

# ✅ Set up client to leverage your free Gemini API configuration via proxy
print("🔄 Initializing OpenAI client wrapper for Gemini endpoint...")
client = OpenAI(
    api_key=CONFIG.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm_model = CONFIG.get("LLM_MODEL", "gpt-4")

# 1. Simulate retrieval (can be real with Pinecone / Graph)
def retrieve_causal_docs(query):
    return [f"Retrieved medical doc for: {query}"], ["cause → condition → effect"]

# 2. Use Gemini to rewrite causal explanation via proxy
def call_rewriting_llm(docs, paths):
    prompt = f"Given the documents:\n{docs}\nand causal paths:\n{paths},\nWrite a structured causal explanation."
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a medically accurate structuring assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.0
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ Generation Error (Rewriter): {e}")
        return f"Structured explanation for paths: {paths}"

# 3. Use Gemini to generate final response via proxy
def call_final_response_llm(query, explanation):
    prompt = f"Query: {query}\nExplanation: {explanation}\nNow provide a final natural language response."
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a user-friendly medical assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.0
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ Generation Error (Final Response): {e}")
        return f"Final response to query: {query}"

# 4. Hallucination detection (Check if core context is omitted or contradicted)
def detect_hallucination(rewritten, final_answer):
    return rewritten.lower() not in final_answer.lower()

# 5. Average graph path depth evaluation
def average_depth(paths):
    return np.mean([p.count("→") + 1 for p in paths]) if paths else 0

# 6. Cosine similarity score between original query variation and rewritten context
def cosine_sim_score(query, rewritten):
    emb1 = encoder.encode(query, normalize_embeddings=True)
    emb2 = encoder.encode(rewritten, normalize_embeddings=True)
    return cos_sim(emb1, emb2).item()

# Unified processing pipeline for measuring strategy rewards
def simulate_cdf_pipeline(refined_query):
    retrieved_docs, causal_paths = retrieve_causal_docs(refined_query)
    rewritten = call_rewriting_llm(retrieved_docs, causal_paths)
    final_answer = call_final_response_llm(refined_query, rewritten)
    hallucinated = detect_hallucination(rewritten, final_answer)

    retrieval_coverage = 1 if retrieved_docs else 0
    depth = average_depth(causal_paths)
    context_relevance = cosine_sim_score(refined_query, rewritten)
    halluc_score = 1 - int(hallucinated)

    # Calculate policy reward mathematically based on pipeline behavior
    reward = (
        0.3 * retrieval_coverage +
        0.3 * depth +
        0.2 * context_relevance +
        0.2 * halluc_score
    )
    return reward

# ----------------------------
# Replicated Gymnasium Environment
# ----------------------------
class QueryRefinementEnv(gym.Env):
    def __init__(self, queries):
        super().__init__()
        self.queries = queries
        self.query_index = 0
        self.embedding_dim = 384

        # Actions: 0 = Expand, 1 = Simplify, 2 = Decompose
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.embedding_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.query_index = (self.query_index + 1) % len(self.queries)
        self.current_query = self.queries[self.query_index]
        emb = encoder.encode(self.current_query, normalize_embeddings=True)
        return emb.astype(np.float32), {}

    def step(self, action):
        query = self.current_query
        strategy = ["Expand", "Simplify", "Decompose"][action]

        if strategy == "Expand":
            refined = f"What are the detailed clinical causes and manifestations of {query}?"
        elif strategy == "Simplify":
            refined = f"What simply causes {query.split()[-1]}?"
        else:  # Decompose
            refined = f"What are the individual physiological subcauses underlying {query}?"

        print(f"🤖 Agent applied action [{strategy}] -> Refined Query: '{refined}'")
        reward = simulate_cdf_pipeline(refined)
        obs = encoder.encode(query, normalize_embeddings=True).astype(np.float32)
        
        # Standard RL steps for query parsing are episodic tasks per step
        terminated = True 
        truncated = False
        
        return obs, reward, terminated, truncated, {}

# ----------------------------
# Execution & Training Loop
# ----------------------------
if __name__ == "__main__":
    # Domain-specific sample queries for evaluation
    medical_queries = [
        "Why do patients develop type 2 diabetes?",
        "What leads to chronic hypertension?",
        "Why does sleep apnea cause daytime fatigue?",
        "What mechanisms trigger myocardial infarction?",
        "Why do individuals experience asthma exacerbation?"
    ]

    print("🏋️ Initializing PPO Agent and Training Environment...")
    env = QueryRefinementEnv(medical_queries)
    model = PPO("MlpPolicy", env, verbose=1)

    total_epochs = 2
    steps_per_epoch = len(medical_queries) * 5
    
    print(font := f"🚀 Training RL Query Refiner policy for {total_epochs} epochs...")
    model.learn(total_timesteps=total_epochs * steps_per_epoch)
    
    model.save("ppo_query_refiner_medical")
    print("💾 Model saved successfully as 'ppo_query_refiner_medical'!")