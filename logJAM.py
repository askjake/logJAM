import os
import argparse
import json
import paramiko
import re
from datetime import datetime
from db.postgres_storer import connect as pg_connect, create_table_if_not_exists, store_parsed_logs
from db.neo4j_storer import Neo4jStorer
from parsers.patterns import parse_log_line
from linking.enricher import enrich_log_data
from ingestion.utils import (
    sftp_list_files, read_remote_file, compute_file_hash, 
    is_file_imported, mark_file_as_imported, create_imported_files_table
)

def extract_log_base_name(file_name):
    """Extracts base name of the log file (removes extension, numbers, and special characters)."""
    base_name = re.sub(r'[^a-zA-Z0-9]', '_', file_name.split('.')[0])  # Replace non-alphanumeric with '_'
    return base_name

def process_log_file(sftp, log_file, pg_conn, neo4j_storer, table_name):
    """Processes a single log file."""
    file_name = os.path.basename(log_file)
    log_base_name = extract_log_base_name(file_name)  #  Get base log name
    rx_id = os.path.dirname(log_file).split('/')[-1]  # Extract RX ID from path
    content = read_remote_file(sftp, log_file)
    if not content:
        return

    file_hash = compute_file_hash(content)  #  Fix: Only pass file content
    if is_file_imported(pg_conn, file_hash):
        print(f"Skipping already imported file: {file_name}")
        return

    lines = content.splitlines()
    logs_to_store = []
    last_valid_timestamp = None  

    for line in lines:
        parsed = parse_log_line(line, last_valid_timestamp, rx_id)  
        if parsed and parsed.get("timestamp"):
            last_valid_timestamp = datetime.fromisoformat(parsed.get("timestamp")) 
            enriched = enrich_log_data(parsed)
            enriched["directory_file"] = file_name 
            logs_to_store.append(enriched)

    # Ensure the correct table name
    table_name = table_name or f"{rx_id}_{log_base_name}"

    create_table_if_not_exists(pg_conn, table_name)
    store_parsed_logs(pg_conn, table_name, logs_to_store, file_name)  

    for log_data in logs_to_store:
        neo4j_storer.store_log_line(log_data)

    mark_file_as_imported(pg_conn, file_name, file_hash)
    print(f"Processed {file_name}")


def main():
    parser = argparse.ArgumentParser(description="Optimized Log Ingestion")
    parser.add_argument('-d', '--directory', type=str, default='/ccshare/logs/smplogs/', help='Remote directory containing log files')
    parser.add_argument('-t', '--table_name', type=str, help='Specify a custom table name for storing logs')

    args = parser.parse_args()

    # Load credentials
    with open("credentials.txt") as f:
        creds = json.load(f)

    neo4j_uri = creds.get("NEO4J_URI")
    neo4j_user = creds.get("NEO4J_USER")
    neo4j_pass = creds.get("NEO4J_PASS")

    # Establish SSH connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(creds["linux_pc"], username=creds["username"], password=creds["password"])
    sftp = ssh.open_sftp()

    # Connect to PostgreSQL
    pg_conn = pg_connect(creds)
    create_imported_files_table(pg_conn)

    # Initialize Neo4j Storer
    neo4j_storer = Neo4jStorer(neo4j_uri, neo4j_user, neo4j_pass)

    # List files in remote directory
    log_files = sftp_list_files(sftp, args.directory)
    print(f"Found {len(log_files)} log files.")

    # Process each log file
    for log_file in log_files:
        log_name = os.path.basename(log_file).split('.')[0]
        rx_id = os.path.dirname(log_file).split('/')[-1]  # Extract RX ID from path
        table_name = args.table_name or f"{rx_id}_{extract_log_base_name(log_name)}"

        process_log_file(sftp, log_file, pg_conn, neo4j_storer, table_name)

    # Cleanup
    sftp.close()
    ssh.close()
    pg_conn.close()
    neo4j_storer.close()
    print("Log ingestion completed.")

if __name__ == "__main__":
    main()
