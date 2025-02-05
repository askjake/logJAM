# schema_manager.py

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
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (l:LogLine) ASSERT l.unique_key IS UNIQUE;")
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (p:Process) ASSERT p.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (s:Session) ASSERT s.sid IS UNIQUE;")
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (c:Connection) ASSERT c.cid IS UNIQUE;")

    def close(self):
        """Closes database connections."""
        self.pg_conn.close()
        self.neo4j_driver.close()

# Usage Example
if __name__ == "__main__":
    from credentials import credentials  # Import credentials from a config file
    schema_manager = SchemaManager(
        {
            "host": credentials["DB_HOST"],
            "database": credentials["DB_NAME"],
            "user": credentials["DB_USER"],
            "password": credentials["DB_PASSWORD"]
        },
        credentials
    )
    schema_manager.create_postgres_tables()
    schema_manager.create_neo4j_constraints()
    schema_manager.close()
