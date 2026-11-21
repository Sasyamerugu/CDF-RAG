import os
import re
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


class DualRetrieverAgent:

    def __init__(self):

        load_dotenv()

        # ---------------------------
        # Pinecone
        # ---------------------------

        api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError("PINECONE_API_KEY not found.")

        self.pc = Pinecone(api_key=api_key)

        self.index = self.pc.Index("cdf-rag-index")

        # ---------------------------
        # Embedding Model
        # ---------------------------

        print("Loading retrieval embedding model...")

        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

        print("Retriever model loaded.")

        # ---------------------------
        # Neo4j
        # ---------------------------

        self.neo4j_driver = GraphDatabase.driver(

            os.getenv("NEO4J_URI", "bolt://localhost:7687"),

            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "")
            )

        )

    # -----------------------------------------------------

    def close(self):

        self.neo4j_driver.close()

    # -----------------------------------------------------

    def retrieve_knowledge(self, state: dict):

        query = state["query"]

        print("\n🔍 Executing Dual Retrieval...")

        # =================================================
        # Semantic Retrieval (Pinecone)
        # =================================================

        query_embedding = self.model.encode(

            query,

            normalize_embeddings=True,

            convert_to_numpy=True

        ).tolist()

        results: Any = self.index.query(

            vector=query_embedding,

            top_k=5,

            include_metadata=True

        )

        semantic_context = []

        matches = getattr(results, "matches", [])

        for match in matches:

            metadata = getattr(match, "metadata", None)

            if metadata:

                text = metadata.get("text")

                if text:

                    semantic_context.append(text)

        print(
            f"  ✅ Pinecone retrieved {len(semantic_context)} semantic documents."
        )

        # =================================================
        # Graph Retrieval (Neo4j)
        # =================================================

        keywords = list(dict.fromkeys(
            re.findall(r"[A-Za-z]+", query.lower())
        ))

        graph_context = []

        cypher = """
        UNWIND $keywords AS keyword

        MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)
        MATCH (q)-[:BELONGS_TO]->(s:Subject)
        MATCH (q)-[:HAS_TOPIC]->(t:Topic)

        WHERE
            toLower(q.question) CONTAINS keyword
            OR toLower(a.text) CONTAINS keyword
            OR toLower(s.name) CONTAINS keyword
            OR toLower(t.name) CONTAINS keyword

        RETURN DISTINCT
            q.question AS question,
            a.text AS answer,
            s.name AS subject,
            t.name AS topic

        LIMIT 20
        """

        try:

            with self.neo4j_driver.session() as session:

                records = session.run(
                    cypher,
                    keywords=keywords
                )

                for record in records:

                    graph_context.append(
                        f"""Question: {record['question']}
Answer: {record['answer']}
Subject: {record['subject']}
Topic: {record['topic']}"""
                    )

        except Exception as e:

            print(f"  ⚠️ Neo4j retrieval failed: {e}")

        graph_context = list(dict.fromkeys(graph_context))

        print(
            f"  ✅ Neo4j retrieved {len(graph_context)} graph records."
        )
        # =================================================
        # Return
        # =================================================

        state["retrieved_docs"] = semantic_context

        state["causal_docs"] = graph_context

        return state