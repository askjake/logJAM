from neo4j import GraphDatabase

class Neo4jStorer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def store_log_line(self, log_data):
        """Stores log lines and their relationships in Neo4j."""
        with self.driver.session() as session:
            session.write_transaction(self._create_nodes_and_relationships, log_data)

    @staticmethod
    def _create_nodes_and_relationships(tx, log_data):
        """Creates nodes and relationships based on log attributes."""
        category = log_data.get("category", "UNKNOWN")
        timestamp = log_data.get("timestamp", "")
        data = log_data.get("data", "")
        event_type = log_data.get("event_type", "")
        stbh = log_data.get("stbh", None)
        cid = log_data.get("cid", None)
        sid = log_data.get("sid", None)

        log_query = """
        MERGE (log:LogLine {unique_key: $uniqueKey})
        ON CREATE SET log.timestamp = $timestamp, log.data = $data, log.event_type = $eventType
        """
        tx.run(log_query, uniqueKey=f"{timestamp}_{category}_{hash(data)}", timestamp=timestamp, data=data, eventType=event_type)

        process_query = "MERGE (p:Process {name: $category})"
        tx.run(process_query, category=category)

        relate_process_log = """
        MATCH (p:Process {name: $category}), (log:LogLine {unique_key: $uniqueKey})
        MERGE (p)-[:GENERATED]->(log)
        """
        tx.run(relate_process_log, category=category, uniqueKey=f"{timestamp}_{category}_{hash(data)}")

        if sid:
            session_query = "MERGE (s:Session {sid: $sid})"
            tx.run(session_query, sid=sid)

            relate_log_session = """
            MATCH (s:Session {sid: $sid}), (log:LogLine {unique_key: $uniqueKey})
            MERGE (log)-[:PART_OF_SESSION]->(s)
            """
            tx.run(relate_log_session, sid=sid, uniqueKey=f"{timestamp}_{category}_{hash(data)}")

        if cid:
            conn_query = "MERGE (c:Connection {cid: $cid})"
            tx.run(conn_query, cid=cid)

            relate_log_conn = """
            MATCH (c:Connection {cid: $cid}), (log:LogLine {unique_key: $uniqueKey})
            MERGE (log)-[:ASSOCIATED_WITH]->(c)
            """
            tx.run(relate_log_conn, cid=cid, uniqueKey=f"{timestamp}_{category}_{hash(data)}")
