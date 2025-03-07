import argparse
import json
import os
import re

import paramiko

from db.neo4j_storer import Neo4jStorer
from db.postgres_storer import connect as pg_connect, create_table_if_not_exists, store_parsed_logs
from parsers.patterns import parse_log_line
from utils import (
    sftp_list_files,
    read_remote_file,
    compute_file_hash,
    is_file_imported,
    mark_file_as_imported
)


###############################################################################
# Helper to extract RxID = "R" followed by 10 digits
###############################################################################
def extract_rxid_from_path(path_str):
    """
    Example: /ccshare/logs/smplogs/R1903684686/2024-08-21
    or /ccshare/logs/smplogs/R1903684686
    => returns R1903684686
    If not found, returns 'UNKNOWN_RXID'
    """
    match = re.search(r'(R\d{10})', path_str)
    if match:
        return match.group(1)
    return "UNKNOWN_RXID"


def extract_log_base_name(file_name):
    base = re.sub(r"\d{2}:\d{2}:\d{2}", "", file_name)
    base = re.sub(r"\d{4}-\d{2}-\d{2}", "", base)
    base = os.path.splitext(base)[0]
    base = re.sub(r"\d+$", "", base)
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", base)
    base = base.strip("_")
    base = re.sub(r"\d+$", "", base)
    base = base.strip("_")
    return base

###############################################################################
# Process a single log file with brand-new short-lived connections
###############################################################################
def process_log_file(sftp, log_file, pg_credentials, import_credentials, neo4j_creds, rx_id):
    from datetime import datetime

    # Create short-lived DB & Neo4j connections
    pg_conn = pg_connect(pg_credentials)
    import_conn = pg_connect(import_credentials)
    neo4j_storer = Neo4jStorer(
        neo4j_creds["NEO4J_URI"],
        neo4j_creds["NEO4J_USER"],
        neo4j_creds["NEO4J_PASS"]
    )
    try:
        file_name = os.path.basename(log_file)
        log_base_name = extract_log_base_name(file_name)

        content = read_remote_file(sftp, log_file)
        if not content:
            return  # Nothing to do

        file_hash = compute_file_hash(content)
        if is_file_imported(import_conn, file_hash):
            print(f"\r[INGEST] Skipping already imported file: {file_name}", end='')
            return

        lines = content.splitlines()
        logs_to_store = []
        last_valid_timestamp = None

        for line in lines:
            parsed = parse_log_line(line, last_valid_timestamp, rx_id)
            if parsed and parsed.get("timestamp"):
                last_valid_timestamp = datetime.fromisoformat(parsed["timestamp"])
                logs_to_store.append(parsed)

        table_name = f"{rx_id}_{log_base_name}"
        create_table_if_not_exists(pg_conn, table_name)
        store_parsed_logs(pg_conn, table_name, logs_to_store, file_name)
        print(f"\r[INGEST] Stored {len(logs_to_store)} logs in table '{table_name}'", flush=True, end='')

        # Insert each log into Neo4j
        for log_data in logs_to_store:
            neo4j_storer.store_log_line(log_data)
            print(f"\r[NEO4J] Processed file {file_name}", flush=True, end='')

        mark_file_as_imported(import_conn, file_name, file_hash)
        print(f"\r[INGEST] Processed file {file_name}", flush=True, end='')
    finally:
        pg_conn.close()
        import_conn.close()
        neo4j_storer.close()


###############################################################################
# Function run by each process => processes an entire subdirectory
###############################################################################
def process_directory(remote_dir, creds, directory_info):
    """
    This function is invoked in a separate process to handle all logs in remote_dir.
    - Creates its own SFTP, its own DB connections, etc.
    - Then calls process_log_file() for each file.
    """
    # Step 1: Connect SFTP for this process
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(creds["linux_pc"], username=creds["username"], password=creds["password"])
    sftp = ssh.open_sftp()

    # Build DB credentials
    pg_credentials = {
        "DB_HOST": creds["DB_HOST"],
        "DB_NAME": directory_info["logs_db_name"],
        "DB_USER": creds["DB_USER"],
        "DB_PASSWORD": creds["DB_PASSWORD"]
    }
    import_credentials = {
        "DB_HOST": creds["DB_HOST"],
        "DB_NAME": directory_info["import_db_name"],
        "DB_USER": creds["DB_USER"],
        "DB_PASSWORD": creds["DB_PASSWORD"]
    }
    neo4j_creds = {
        "NEO4J_URI": creds["NEO4J_URI"],
        "NEO4J_USER": creds["NEO4J_USER"],
        "NEO4J_PASS": creds["NEO4J_PASS"]
    }

    # Step 2: Extract the RxID from remote_dir
    rx_id = extract_rxid_from_path(remote_dir)
    #print(f"[INGEST] For directory '{remote_dir}', extracted RxID: {rx_id}")

    # Step 3: List all log files
    log_files = sftp_list_files(sftp, remote_dir)

    file_count = 0
    for log_file in log_files:
        process_log_file(sftp, log_file, pg_credentials, import_credentials, neo4j_creds, rx_id)
        file_count += 1

    # Cleanup
    sftp.close()
    ssh.close()
    return rx_id, file_count


def list_remote_subdirs(sftp, remote_path):
    """List only subdirectories in the remote path."""
    try:
        dirs = []
        for attr in sftp.listdir_attr(remote_path):
            if attr.st_mode & 0o040000:  # directory bit
                dirs.append(f"{remote_path}/{attr.filename}")
        return dirs
    except Exception as e:
        print(f"[ERROR] Failed to list subdirectories for {remote_path}: {e}")
        return []

###############################################################################
# MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Optimized Log Ingestion (Parallel per directory).")
    parser.add_argument('-d', '--directory', type=str, default='/ccshare/logs/smplogs/',
                        help='Remote "root" directory containing multiple subdirectories of log files')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='Number of parallel processes to handle subdirectories')
    args = parser.parse_args()

    # Load credentials
    with open("credentials.txt") as f:
        creds = json.load(f)

    directory_info = {
        "logs_db_name": creds["DB_NAME"],
        "import_db_name": creds["IMPORT_DB"]
    }

    # Pre-check for subdirectories
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(creds["linux_pc"], username=creds["username"], password=creds["password"])
    sftp_main = ssh.open_sftp()

    subdirs = list_remote_subdirs(sftp_main, args.directory)
    sftp_main.close()
    ssh.close()

    if not subdirs:
        print("[WARN] No subdirectories found. Checking logs in root directory instead.")
        subdirs = [args.directory]

    print(f"[INGEST] Found {len(subdirs)} directories to process under {args.directory}.")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_dir = {
            executor.submit(process_directory, d, creds, directory_info): d
            for d in subdirs
        }
        for future in as_completed(future_to_dir):
            directory = future_to_dir[future]
            try:
                rx_id, fcount = future.result()
                results.append((rx_id, fcount))
                #print(f"[INGEST] Completed directory {directory} => {fcount} files. RxID={rx_id}")
            except Exception as ex:
                print(f"[ERROR] Failed processing {directory}: {ex}")

    print("[INGEST] Done parallel ingestion of subdirectories.")
    for rx_id, fcount in results:
        print(f"   - Directory: {rx_id}, Files Ingested: {fcount}")

    print("[INGEST] Log ingestion completed. Now run `enricher.py` for enrichment.")


if __name__ == "__main__":
    main()
