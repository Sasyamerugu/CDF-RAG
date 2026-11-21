import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class Neo4jIngestor:

    def __init__(self):

        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "dhanush_password")
            )
        )

    def close(self):
        self.driver.close()

    def add_question(self, row):

        query = """
        MERGE (q:Question {id:$id})

        SET
            q.question = $question,
            q.option_a = $option_a,
            q.option_b = $option_b,
            q.option_c = $option_c,
            q.option_d = $option_d,
            q.explanation = $explanation

        MERGE (a:Answer {text:$answer})
        MERGE (s:Subject {name:$subject})
        MERGE (t:Topic {name:$topic})

        MERGE (q)-[:HAS_ANSWER]->(a)
        MERGE (q)-[:BELONGS_TO]->(s)
        MERGE (q)-[:HAS_TOPIC]->(t)

        MERGE (a)-[:BELONGS_TO]->(s)
        MERGE (a)-[:RELATED_TO]->(t)
        """

        try:

            with self.driver.session() as session:

                session.run(

                    query,

                    id=int(row["id"]),

                    question=row["question"],

                    option_a=row["option_a"],
                    option_b=row["option_b"],
                    option_c=row["option_c"],
                    option_d=row["option_d"],

                    explanation=row["explanation"],

                    answer=row["answer"],

                    subject=row["subject"],

                    topic=row["topic"]
                )

            print(f"✅ Added Question {row['id']}")

        except Exception as e:

            print(f"⚠️ Question {row['id']} failed: {e}")


if __name__ == "__main__":

    csv_path = "processed_medmcqa.csv"

    if not os.path.exists(csv_path):

        print(f"{csv_path} not found.")

    else:

        df = pd.read_csv(csv_path)

        df = df.fillna("")

        db = Neo4jIngestor()

        print("🚀 Uploading MedMCQA to Neo4j...")

        for _, row in df.iterrows():

            db.add_question(row)

        db.close()

        print("\n💾 Neo4j ingestion complete!")
        print("🌐 Open http://localhost:7474")