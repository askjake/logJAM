# ingestion/log_ingest.py
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
from utils import (
    sftp_list_files, read_remote_file, compute_file_hash, 
    is_file_imported, mark_file_as_imported, create_imported_files_table
)
from analysis.anomaly_detection.autoencoder import run_autoencoder_analysis
from analysis.anomaly_detection.lstm_anomaly import run_lstm_analysis

def extract_log_base_name(file_name):
    # Remove extension
    base = file_name.split('.')[0]
    # Replace non-alphanumeric (except underscore) with underscore
    base = re.sub(r'[^a-zA-Z0-9_]', '_', base)
    # Remove any timestamp (HH:MM:SS) and date (YYYY-MM-DD) patterns
    base = base.strip('_')
    base = re.sub(r'^(?:\d+_)+', '', base)
        
    parts = base.split('_', 1)
    if len(parts) == 2:
        base_name = parts[1]
    else:
        base_name = base
    print(f"[DEBUG] Extracted table base name: {base_name}")
    return base_name

def process_log_file(sftp, log_file, pg_conn, neo4j_storer, table_name):
    """Processes a single log file."""
    file_name = os.path.basename(log_file)
    print(f"Processing file: {file_name}")
    content = read_remote_file(sftp, log_file)
    if not content:
        return

    file_hash = compute_file_hash(content)
    if is_file_imported(pg_conn, file_hash):
        print(f"Skipping already imported file: {file_name}")
        return

    lines = content.splitlines()
    logs_to_store = []
    last_valid_timestamp = None  

    for line in lines:
        # Pass rx_id as None here because the RXID is used only to build the table name.
        parsed = parse_log_line(line, last_valid_timestamp, None)
        if parsed and parsed.get("timestamp"):
            last_valid_timestamp = datetime.fromisoformat(parsed.get("timestamp"))
            enriched = enrich_log_data(parsed)
            enriched["directory_file"] = file_name 
            logs_to_store.append(enriched)

    # Do not recalc table_name here; use the one provided.
    print(f"[DEBUG] Processed table name: {table_name}")

    create_table_if_not_exists(pg_conn, table_name)
    store_parsed_logs(pg_conn, table_name, logs_to_store, file_name)

    # Pass the list of enriched logs to the Neo4j storer.
    if logs_to_store:
        neo4j_storer.store_log_lines(logs_to_store)

    mark_file_as_imported(pg_conn, file_name, file_hash)
    print(f"Processed {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Optimized Log Ingestion")
    parser.add_argument('-d', '--directory', type=str, default='/ccshare/logs/smplogs/',
                        help='Remote directory containing log files')
    parser.add_argument('-t' '--table_name', type=str, help='Specify a custom table name for storing logs')
    parser.add_argument('-a', '--analysis', type=str, choices=['none','autoencoder','lstm'], default='none',
                        help='Which analysis to run after ingestion')
    parser.add_argument('-ta', '--table_for_analysis', type=str, 
                        help='Which table name to run anomaly detection on (if not provided, uses last table).')
    args = parser.parse_args()

    # Load credentials
    with open("credentials.txt") as f:
        creds = json.load(f)

    neo4j_uri = creds.get("NEO4J_URI")
    neo4j_user = creds.get("NEO4J_USER")
    neo4j_pass = creds.get("NEO4J_PASS")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(creds["linux_pc"], username=creds["username"], password=creds["password"])
    sftp = ssh.open_sftp()

    pg_conn = pg_connect(creds)
    create_imported_files_table(pg_conn)

    neo4j_storer = Neo4jStorer(neo4j_uri, neo4j_user, neo4j_pass)

    log_files = sftp_list_files(sftp, args.directory)
    print(f"Found {len(log_files)} log files.")

    rx_id = os.path.basename(args.directory.rstrip('/'))
    last_table_name = None  # We'll store the last table we used if user doesn't specify one

    for log_file in log_files:
        file_name_no_ext = os.path.basename(log_file).split('.')[0]
        cleaned_log_name = extract_log_base_name(file_name_no_ext)
        table_name = args.t__table_name or f"{rx_id}_{cleaned_log_name}"
        last_table_name = table_name
        process_log_file(sftp, log_file, pg_conn, neo4j_storer, table_name)

    sftp.close()
    ssh.close()
    
    if not args.analysis:
        print("No analyses requested. Skipping anomaly detection.")
    else:
        target_table = args.table_for_analysis or last_table_name
        if not target_table:
            print("No table was created. Skipping analysis.")
        else:
            # Check if user wants autoencoder
            if 'autoencoder' in args.analysis:
                run_autoencoder_analysis(pg_conn, target_table, latent_dim=8, epochs=10)

            # Check if user wants LSTM
            if 'lstm' in args.analysis:
                run_lstm_analysis(pg_conn, target_table, seq_len=10, hidden_size=16, epochs=10)


    pg_conn.close()
    neo4j_storer.close()
    print("Log ingestion completed.")

if __name__ == "__main__":
    main()