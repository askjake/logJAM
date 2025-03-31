#!/usr/bin/env python3

## log_ingest.py ##

import argparse
import json
import os
import re
import paramiko
import multiprocessing
import sys
import time
from parsers.var import parse_var
from parsers.ktrap import parse_ktrap_file
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
# Function to offload Neo4j storage in a separate process (less verbose now)
###############################################################################
def store_logs_in_neo4j(logs_to_store, neo4j_creds):
    from db.neo4j_storer import Neo4jStorer
    neo4j_storer = Neo4jStorer(
        neo4j_creds["NEO4J_URI"],
        neo4j_creds["NEO4J_USER"],
        neo4j_creds["NEO4J_PASS"]
    )
    try:
        for log_data in logs_to_store:
            neo4j_storer.store_log_line(log_data)
        print(f"[NEO4J] Stored {len(logs_to_store)} logs.")
    except Exception as e:
        print(f"[NEO4J ERROR] {e}")
    finally:
        neo4j_storer.close()

###############################################################################
# [NEW SCHEMA GROOMING] – Ensure table schema is current
###############################################################################
def groom_table_schema(pg_conn, table_name):
    print(f"[INFO] Grooming table schema for '{table_name}'...")
    create_table_if_not_exists(pg_conn, table_name)
    # Add additional schema grooming here as needed.

###############################################################################
# Process a single log file with improved error handling and progress update
###############################################################################


def process_log_file(sftp, log_file, pg_credentials, import_credentials, neo4j_creds, rx_id):
    from datetime import datetime

    pg_conn = pg_connect(pg_credentials)
    import_conn = pg_connect(import_credentials)
    neo4j_process = None

    try:
        file_name = os.path.basename(log_file)
        print(f"--> Processing file: {file_name}")
        log_base_name = extract_log_base_name(file_name)

        content = read_remote_file(sftp, log_file)
        if not content:
            print(f"[WARN] {file_name}: No content.")
            return None

        file_hash = compute_file_hash(content)
        if is_file_imported(import_conn, file_hash):
            print(f"[SKIP] {file_name} already imported.")
            return None

        # Determine which parser to use
        if var_detected(content):
            print(f"[INFO] Detected VAR log format in {file_name}.")
            logs_to_store = parse_var(content)
        elif "k_trap" in file_name.lower():
            logs_to_store = parse_ktrap_file(content, rx_id)
        else:
            # Use the default parser
            lines = content.splitlines()
            logs_to_store = []
            last_valid_timestamp = None
            for line in lines:
                parsed = parse_log_line(line, last_valid_timestamp, rx_id)
                if parsed and parsed.get("timestamp"):
                    last_valid_timestamp = datetime.fromisoformat(parsed["timestamp"])
                    logs_to_store.append(parsed)

        table_name_full = f"{rx_id}_{log_base_name}"
        print(f"[INFO] Storing logs in table: {table_name_full}")
        create_table_if_not_exists(pg_conn, table_name_full)
        store_parsed_logs(pg_conn, table_name_full, logs_to_store, file_name)
        print(f"[INFO] {len(logs_to_store)} logs stored in '{table_name_full}'")

        # Mark file as imported; protect against aborted transactions
        try:
            mark_file_as_imported(import_conn, file_name, file_hash)
        except Exception as e:
            import_conn.rollback()
            print(f"[ERROR] Could not mark file '{file_name}' as imported: {e}")

        neo4j_process = multiprocessing.Process(
            target=store_logs_in_neo4j,
            args=(logs_to_store, neo4j_creds)
        )
        neo4j_process.start()

        groom_table_schema(pg_conn, table_name_full)

        return neo4j_process

    except Exception as e:
        print(f"[ERROR] Exception processing {file_name}: {e}")
    finally:
        pg_conn.close()
        import_conn.close()


def var_detected(content):
    # We assume new log lines start with a date like "2025-03-27" followed by a space,
    # then a time with milliseconds and then two numbers (pid and tid).
    first_line = content.splitlines()[0].strip() if content else ""
    return re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+', first_line) is not None
###############################################################################
# Process all files in a directory with a progress meter
###############################################################################
def process_directory(remote_dir, creds, directory_info):
    #print(f"[INFO] Processing directory: {remote_dir}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(creds["linux_pc"], username=creds["username"], password=creds["password"])
    except Exception as e:
        print(f"[ERROR] SSH connection failed: {e}")
        sys.exit(1)
    sftp = ssh.open_sftp()

    pg_credentials = {
        "DB_HOST": creds["DB_HOST"],
        "DB_NAME": directory_info["logs_db_name"],
        "DB_USER": creds["DB_USER"],
        "DB_PASSWORD": creds["DB_PASSWORD"]
    }
    if directory_info.get("use_good"):
        import_credentials = {
            "DB_HOST": creds["DB_HOST"],
            "DB_NAME": creds["HAPPY_PATH_DB"],
            "DB_USER": creds["HAPPY_PATH_USER"],
            "DB_PASSWORD": creds["HAPPY_PATH_PASSWORD"]
        }
    else:
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

    rx_id = extract_rxid_from_path(remote_dir)
    log_files = sftp_list_files(sftp, remote_dir)
    total_files = len(log_files)
    print(f"[INFO] {total_files} files found in {remote_dir}")

    neo4j_processes = []
    for idx, log_file in enumerate(log_files, start=1):
        proc = process_log_file(sftp, log_file, pg_credentials, import_credentials, neo4j_creds, rx_id)
        if proc:
            neo4j_processes.append(proc)
        # Simple progress meter update (overwrites the same line)
        sys.stdout.write(f"[PROGRESS] Processed {idx}/{total_files} files in {rx_id}")
        sys.stdout.flush()
        time.sleep(0.1)  # Optional small delay to make updates readable

    print()  # Newline after progress meter
    for proc in neo4j_processes:
        proc.join()
        print(f"[INFO] Neo4j process {proc.pid} completed.")

    sftp.close()
    ssh.close()
    return rx_id, total_files

def list_remote_subdirs(sftp, remote_path):
    try:
        dirs = []
        for attr in sftp.listdir_attr(remote_path):
            if attr.st_mode & 0o040000:
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
                        help='Remote root directory containing subdirectories of log files')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='Number of parallel processes to handle subdirectories')
    parser.add_argument("--good", action="store_true",
                        help="Ingest logs as 'good' and store in HAPPY_PATH DB.")
    args = parser.parse_args()

    with open("credentials.txt") as f:
        creds = json.load(f)

    if args.good:
        for key in ["HAPPY_PATH_DB", "HAPPY_PATH_USER", "HAPPY_PATH_PASSWORD"]:
            if key not in creds:
                print(f"[ERROR] Missing key '{key}' in credentials.txt for good logs ingestion.")
                sys.exit(1)

    directory_info = {
        "logs_db_name": creds["DB_NAME"],
        "import_db_name": creds["IMPORT_DB"]
    }
    if args.good:
        directory_info["logs_db_name"] = creds["HAPPY_PATH_DB"]
        directory_info["import_db_name"] = creds["HAPPY_PATH_DB"]
        directory_info["use_good"] = True

    print(f"[INFO] Directory info: {directory_info}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(creds["linux_pc"], username=creds["username"], password=creds["password"])
    except Exception as e:
        print(f"[ERROR] SSH connection failed: {e}")
        sys.exit(1)
    sftp_main = ssh.open_sftp()
    subdirs = list_remote_subdirs(sftp_main, args.directory)
    sftp_main.close()
    ssh.close()

    if not subdirs:
        print("[WARN] No subdirectories found. Using root directory instead.")
        subdirs = [args.directory]

    total_dirs = len(subdirs)
    print(f"[INGEST] Found {total_dirs} directories to process under {args.directory}.")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = []
    dir_counter = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_dir = {executor.submit(process_directory, d, creds, directory_info): d for d in subdirs}
        for future in as_completed(future_to_dir):
            directory = future_to_dir[future]
            try:
                rx_id, fcount = future.result()
                results.append((rx_id, fcount))
            except Exception as ex:
                print(f"[ERROR] Failed processing {directory}: {ex}")
            dir_counter += 1
            sys.stdout.write(f"[OVERALL PROGRESS] Processed {dir_counter}/{total_dirs} directories")
            sys.stdout.flush()
    print()  # Newline

    print("[INGEST] Directory processing complete:")
    for rx_id, fcount in results:
        print(f"   - {rx_id}: {fcount} files ingested")
    print("[INGEST] Log ingestion completed.")

if __name__ == "__main__":
    main()
