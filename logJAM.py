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
    sftp_list_files,
    read_remote_file,
    compute_file_hash,
    is_file_imported,
    mark_file_as_imported,
    create_imported_files_table,
    connect_import_db
)

def extract_log_base_name(file_name):
    """
    Extracts a base name of the log file by:
      1. Removing HH:MM:SS timestamps (e.g. '05:11:30')
      2. Removing YYYY-MM-DD dates (e.g. '2025-01-15')
      3. Splitting off the file extension
      4. Removing trailing digits (e.g. '1234' at the end)
      5. Replacing non-alphanumeric (except underscores) with underscores
      6. Stripping leading/trailing underscores
    """
    base = re.sub(r"\d{2}:\d{2}:\d{2}", "", file_name)   # remove HH:MM:SS
    base = re.sub(r"\d{4}-\d{2}-\d{2}", "", base)        # remove YYYY-MM-DD
    base = os.path.splitext(base)[0]                    # remove extension
    base = re.sub(r"\d+$", "", base)                    # remove trailing digits
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", base)         # non-alphanumeric => underscore
    base = base.strip("_")                              # strip underscores
    return base

def process_log_file(sftp, log_file, pg_conn, import_conn, neo4j_storer, rx_id):
    """
    Processes a single log file:
      - read content via SFTP
      - check if file is imported (using import_conn)
      - parse & store logs in DB_NAME (using pg_conn)
      - store relationships in Neo4j
      - mark file as imported in import_db (using import_conn)
    """
    file_name = os.path.basename(log_file)
    log_base_name = extract_log_base_name(file_name)
    content = read_remote_file(sftp, log_file)
    if not content:
        return

    file_hash = compute_file_hash(content)
    if is_file_imported(import_conn, file_hash):
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

    table_name = f"{rx_id}_{log_base_name}"
    create_table_if_not_exists(pg_conn, table_name) 
    store_parsed_logs(pg_conn, table_name, logs_to_store, file_name)

    # Store in Neo4j
    for log_data in logs_to_store:
        neo4j_storer.store_log_line(log_data)

    # Mark file as imported in import_db
    mark_file_as_imported(import_conn, file_name, file_hash)
    print(f"Processed {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Optimized Log Ingestion")
    parser.add_argument('-d', '--directory', type=str, default='/ccshare/logs/smplogs/',
                        help='Remote directory containing log files')
    args = parser.parse_args()

    # Load credentials
    with open("credentials.txt") as f:
        creds = json.load(f)

    # We have two databases: DB_NAME for logs, IMPORT_DB for imported_files
    logs_db_name = creds["DB_NAME"]
    import_db_name = creds["IMPORT_DB"]

    neo4j_uri = creds["NEO4J_URI"]
    neo4j_user = creds["NEO4J_USER"]
    neo4j_pass = creds["NEO4J_PASS"]

    # SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(creds["linux_pc"], username=creds["username"], password=creds["password"])
    sftp = ssh.open_sftp()

    # Connect to logs DB
    log_conn = pg_connect({
        "DB_HOST": creds["DB_HOST"],
        "DB_NAME": logs_db_name,
        "DB_USER": creds["DB_USER"],
        "DB_PASSWORD": creds["DB_PASSWORD"]
    })

    # Connect to import DB
    import_conn = pg_connect({
        "DB_HOST": creds["DB_HOST"],
        "DB_NAME": import_db_name,
        "DB_USER": creds["DB_USER"],
        "DB_PASSWORD": creds["DB_PASSWORD"]
    })

    # Create imported_files table in the import_db
    create_imported_files_table(import_conn)

    # Initialize Neo4j
    neo4j_storer = Neo4jStorer(neo4j_uri, neo4j_user, neo4j_pass)

    log_files = sftp_list_files(sftp, args.directory)
    print(f"Found {len(log_files)} log files.")
    rx_id = os.path.basename(args.directory)

    for log_file in log_files:
        process_log_file(
            sftp, log_file,
            log_conn, import_conn,
            neo4j_storer,
            rx_id
        )

    # Cleanup
    sftp.close()
    ssh.close()
    log_conn.close()
    import_conn.close()
    neo4j_storer.close()
    print("Log ingestion completed.")

if __name__ == "__main__":
    main()
