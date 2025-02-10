import os
import json
import psycopg2
from neo4j import GraphDatabase

class SchemaManager:
    """Manages schema creation and migrations for PostgreSQL and Neo4j."""

    def __init__(self, pg_credentials, neo4j_credentials):
        self.pg_conn = psycopg2.connect(**pg_credentials)
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_credentials["NEO4J_URI"],
            auth=(neo4j_credentials["NEO4J_USER"], neo4j_credentials["NEO4J_PASS"])
        )

    def create_postgres_tables(self):
        """Creates necessary PostgreSQL tables if they don't exist."""
        with self.pg_conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS log_entries (
                    id SERIAL PRIMARY KEY,
                    directory_file TEXT NOT NULL,
                    category TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    file_line TEXT,
                    function TEXT,
                    data TEXT,
                    data_hash TEXT UNIQUE,
                    message TEXT
                );
            """)
            self.pg_conn.commit()

    def create_neo4j_constraints(self):
        """Creates necessary constraints in Neo4j."""
        with self.neo4j_driver.session() as session:
            # Create a unique constraint for LogLine.unique_key
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (l:LogLine) REQUIRE l.unique_key IS UNIQUE
            """)

            # Create a unique constraint for Process.name
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (p:Process) REQUIRE p.name IS UNIQUE
            """)

            # Create a unique constraint for Session.sid
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (s:Session) REQUIRE s.sid IS UNIQUE
            """)

            # Create a unique constraint for Connection.cid
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (c:Connection) REQUIRE c.cid IS UNIQUE
            """)

    def close(self):
        """Closes database connections."""
        self.pg_conn.close()
        self.neo4j_driver.close()


if __name__ == "__main__":
    # Determine the project root directory (assumes schema_manager.py is in <project>/db/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    credentials_file = os.path.join(base_dir, "credentials.txt")

    # Load credentials from the JSON file
    with open(credentials_file, "r") as f:
        credentials = json.load(f)

    # Build PostgreSQL credentials dictionary.
    pg_credentials = {
        "host": credentials["DB_HOST"],
        "database": credentials["DB_NAME"],
        "user": credentials["DB_USER"],
        "password": credentials["DB_PASSWORD"]
    }

    # Use the same credentials for Neo4j (as long as keys match)
    neo4j_credentials = credentials

    # Initialize and use the SchemaManager
    schema_manager = SchemaManager(pg_credentials, neo4j_credentials)
    schema_manager.create_postgres_tables()
    schema_manager.create_neo4j_constraints()
    schema_manager.close()
