from neo4j import GraphDatabase
import shutil


def real_time_status_update(category, data):
    """ Updates the status message every 0.5s while logs are being processed. """
    for _ in range(10):  # Adjust iterations based on processing speed
        term_width = shutil.get_terminal_size((50, 20)).columns
        output = f"[NEO4J DEBUG] Processing... Category: {category} -> {data}"

        if len(output) > term_width:
            output = output[:term_width-3] + "..."
        print(f"\r{output}", flush=True, end='')



class Neo4jStorer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        #print("[NEO4J DEBUG] Initialized Neo4j driver.")

    def close(self):
        self.driver.close()
        #print("[NEO4J DEBUG] Closed Neo4j driver.")

    def store_log_line(self, log_data):
        """Stores a single log line (kept for backwards compatibility)."""
        #print(f"[NEO4J DEBUG] Storing single log entry")
        with self.driver.session() as session:
            session.execute_write(self._create_nodes_and_relationships, log_data)

    def store_log_lines(self, log_data_list, batch_size=1000):
        """
        Stores log lines in batches to improve efficiency.
        :param log_data_list: List of log_data dictionaries.
        :param batch_size: Number of log lines to process in a single transaction.
        """
        if not isinstance(log_data_list, list):
            raise TypeError(f"❌ Expected list of dictionaries, but got {type(log_data_list)}")

        total = len(log_data_list)
        total_batches = (total + batch_size - 1) // batch_size
        print(f"\r[NEO4J DEBUG] Preparing to insert {total} log lines in {total_batches} batches of {batch_size}...", end='')

        # Pre-compute the unique key for each log and ensure required fields.
        for log in log_data_list:
            if not isinstance(log, dict):
                raise TypeError(f"❌ Expected dictionary in list, but got {type(log)}")
            if not log.get("category"):
                print(f"\r[NEO4J WARN] Log entry missing category, defaulting to 'UNKNOWN': {log}", end='')
                log["category"] = "UNKNOWN"
            category = log.get("category", "UNKNOWN")
            timestamp = log.get("timestamp", "")
            data = log.get("data", "")
            log["unique_key"] = f"{timestamp}_{category}_{hash(data)}"

        for i in range(0, total, batch_size):
            batch = log_data_list[i:i+batch_size]
            with self.driver.session() as session:
                session.execute_write(self._create_nodes_and_relationships_batch, batch)
            print(f"[NEO4J DEBUG] Batch {i//batch_size+1}/{total_batches} : Inserted {len(batch)} log lines.", end='')


    @staticmethod
    def _create_nodes_and_relationships(tx, log_data):
        """Stores a single log entry in Neo4j with real-time progress updates."""
        category = log_data.get("category", "UNKNOWN")
        timestamp = log_data.get("timestamp", "")
        data = log_data.get("data", "")
        event_type = log_data.get("event_type", "")
        sid = log_data.get("sid", None)
        cid = log_data.get("cid", None)

        unique_key = f"{timestamp}_{category}_{hash(data)}"

        # Background update indicator
        # real_time_status_update(category, data)
        print(f"\r[NEO4J DEBUG] Processing... Category: {category} ", end='')

        log_query = """
        MERGE (log:LogLine {unique_key: $uniqueKey})
        ON CREATE SET log.timestamp = $timestamp, log.data = $data, log.event_type = $eventType
        """
        tx.run(log_query, uniqueKey=unique_key, timestamp=timestamp, data=data, eventType=event_type)

        process_query = "MERGE (p:Process {name: $category})"
        tx.run(process_query, category=category)

        relate_process_log = """
        MATCH (p:Process {name: $category}), (log:LogLine {unique_key: $uniqueKey})
        MERGE (p)-[:GENERATED]->(log)
        """
        tx.run(relate_process_log, category=category, uniqueKey=unique_key)

        if sid:
            session_query = "MERGE (s:Session {sid: $sid})"
            tx.run(session_query, sid=sid)
            relate_log_session = """
            MATCH (s:Session {sid: $sid}), (log:LogLine {unique_key: $uniqueKey})
            MERGE (log)-[:PART_OF_SESSION]->(s)
            """
            tx.run(relate_log_session, sid=sid, uniqueKey=unique_key)

        if cid:
            conn_query = "MERGE (c:Connection {cid: $cid})"
            tx.run(conn_query, cid=cid)
            relate_log_conn = """
            MATCH (c:Connection {cid: $cid}), (log:LogLine {unique_key: $uniqueKey})
            MERGE (log)-[:ASSOCIATED_WITH]->(c)
            """
            tx.run(relate_log_conn, cid=cid, uniqueKey=unique_key)


    @staticmethod
    def _create_nodes_and_relationships_batch(tx, batch):
        """
        Batch version that uses UNWIND to process multiple log lines.
        Each log in the batch is expected to have a pre-computed "unique_key".
        """
        print(f"\r[NEO4J DEBUG] Inserting batch of {len(batch)} log lines...", end='')
        query = """
        UNWIND $batch AS log
        MERGE (l:LogLine {unique_key: log.unique_key})
        ON CREATE SET l.timestamp = log.timestamp, l.data = log.data, l.event_type = log.event_type
        MERGE (p:Process {name: log.category})
        MERGE (p)-[:GENERATED]->(l)
        FOREACH (_ IN CASE WHEN log.sid IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s:Session {sid: log.sid})
            MERGE (l)-[:PART_OF_SESSION]->(s)
        )
        FOREACH (_ IN CASE WHEN log.cid IS NOT NULL THEN [1] ELSE [] END |
            MERGE (c:Connection {cid: log.cid})
            MERGE (l)-[:ASSOCIATED_WITH]->(c)
        )
        """
        tx.run(query, batch=batch)
        print(f"[NEO4J DEBUG] Batch Insert Complete.")
