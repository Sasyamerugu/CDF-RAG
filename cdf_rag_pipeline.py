import argparse
from typing import TypedDict
from langgraph.graph import StateGraph, END

# ===== MYNE Imports =====
from retrieval.query_refiner_rl import QueryRefinementAgent
from data.Dual_retriver import DualRetrieverAgent
from agents.knowledge_rewriter_rl import KnowledgeRewritingAgent
from agents.llm_inference_response_generator import LLMResponseGenerator
from agents.hallucination_detector_rl import HallucinationCorrectionAgent


# ------------------------------------------------------------------
# Wrapper for MYNE Response Generator
# ------------------------------------------------------------------
class LLMResponseAgent:

    def __init__(self):
        self.generator = LLMResponseGenerator()

    def generate_validated_response(self, state):

        response = self.generator.generate(
            knowledge=state.get("rewritten_knowledge", ""),
            original_query=state["query"]
        )

        return {
            "response": response
        }

# ------------------------------------------------------------------
# Original State (kept almost identical)
# ------------------------------------------------------------------
class CDFRAGState(TypedDict):

    query: str

    retrieved_docs: list[str]

    causal_docs: list[str]

    rewritten_knowledge: str

    response: str

    is_hallucination: bool


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(query: str):

    print("\nInitializing CDF-RAG pipeline with LangGraph and RL")

    builder = StateGraph(CDFRAGState)

    retriever = DualRetrieverAgent()

    builder.add_node(
        "refine_query",
        QueryRefinementAgent().optimize_query
    )

    builder.add_node(
        "retrieve_knowledge",
        retriever.retrieve_knowledge
    )

    builder.add_node(
        "optimize_rewriting",
        KnowledgeRewritingAgent().optimize_rewriting
    )

    builder.add_node(
        "generate_validated_response",
        LLMResponseAgent().generate_validated_response
    )

    builder.add_node(
        "correct_hallucinations",
        HallucinationCorrectionAgent().correct_hallucinations
    )

    builder.set_entry_point("refine_query")

    builder.add_edge(
        "refine_query",
        "retrieve_knowledge"
    )

    builder.add_edge(
        "retrieve_knowledge",
        "optimize_rewriting"
    )

    builder.add_edge(
        "optimize_rewriting",
        "generate_validated_response"
    )

    builder.add_edge(
        "generate_validated_response",
        "correct_hallucinations"
    )

    builder.add_edge(
        "correct_hallucinations",
        END
    )

    graph = builder.compile()

    state = {
        "query": query
    }

    try:

        final_state = graph.invoke(state)

        print("\n========================================")
        print("Final Response\n")
        print(final_state["response"])

        if final_state.get("is_hallucination"):
            print("\n⚠️  Hallucination detected by verification agent.")

        print("========================================\n")

    finally:

        retriever.close()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        nargs="?",
        help="Medical question"
    )

    args = parser.parse_args()

    if args.query:

        main(args.query)

    else:

        while True:

            query = input(
                "\n🩺 Enter your medical question (or 'q' to quit):\n> "
            )

            if query.lower() in {
                "q",
                "quit",
                "exit"
            }:
                print("Shutting down pipeline...")
                break

            if not query.strip():
                continue

            main(query)