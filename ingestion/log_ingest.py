import os
import argparse
import json
import paramiko

from db.postgres_storer import connect as pg_connect, create_table_if_not_exists, store_parsed_logs
from db.neo4j_storer import Neo4jStorer
from parsers.patterns import parse_log_line
from linking.enricher import enrich_log_data
from ingestion.utils import sftp_list_files, read_remote_file, compute_file_hash, is_file_imported, mark_file_as_imported, create_imported_files_table

def main():
    parser = argparse.ArgumentParser(description="Log Ingestion")
    parser.add_argument('-d', '--directory', type=str, default='/ccshare/logs/smplogs/', help='Remote directory containing log files')
    args = parser.parse_args()

    # Load credentials
    with open("credentials.json") as f:
        creds = json.load(f)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(creds["linux_pc"], username=creds["username"], password=creds["password"])
    sftp = ssh.open_sftp()

    # PostgreSQL setup
    pg_conn = pg_connect(creds)
    create_imported_files_table(pg_conn)  # same logic from your old script

    # (Optional) Neo4j setup
    neo4j_storer = Neo4jStorer(creds["NEO4J_URI"], creds["NEO4J_USER"], creds["NEO4J_PASS"])

    # List files in remote directory
    log_files = sftp_list_files(sftp, args.directory)
    print(f"Found {len(log_files)} log files.")

    for log_file in log_files:
        file_name = os.path.basename(log_file)
        content = read_remote_file(sftp, log_file)
        if not content:
            continue

        file_hash = compute_file_hash(content)
        if is_file_imported(pg_conn, file_hash):
            print(f"Skipping already imported file: {file_name}")
            continue

        # parse lines
        lines = content.splitlines()
        logs_to_store = []
        last_valid_timestamp = None
        for line in lines:
            parsed = parse_log_line(line, last_valid_timestamp)
            if parsed and parsed.get("timestamp"):
                # update last_valid_timestamp
                last_valid_timestamp = ...
                # enrich
                enriched = enrich_log_data(parsed)
                logs_to_store.append(enriched)
        
        # store in Postgres
        # table_name can be derived from e.g. rx_id or a default
        table_name = "my_log_table"
        create_table_if_not_exists(pg_conn, table_name)
        store_parsed_logs(pg_conn, table_name, logs_to_store, file_name)

        # store in Neo4j
        for log_data in logs_to_store:
            neo4j_storer.store_log_line(log_data)

        mark_file_as_imported(pg_conn, file_name, file_hash)

    sftp.close()
    ssh.close()
    pg_conn.close()
    neo4j_storer.close()

if __name__ == "__main__":
    main()
