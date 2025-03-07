# ingestion/utils.py

import os
import stat
import gzip
import hashlib
import psycopg2


def create_database_if_not_exists(admin_host, admin_user, admin_password, target_db):
    """
    Connects to the 'postgres' default DB on admin_host using admin_user/admin_password.
    Checks if 'target_db' exists, if not creates it.
    admin_user must have CREATEDB privileges or be superuser.
    """
    try:
        # Connect to the default 'postgres' db
        admin_conn = psycopg2.connect(
            host=admin_host,
            dbname="postgres",
            user=admin_user,
            password=admin_password
        )
        admin_conn.autocommit = True
        with admin_conn.cursor() as cur:
            # check if DB exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (target_db,))
            exists = cur.fetchone()
            if not exists:
                print(f"[AUTO-DB] Database '{target_db}' doesn't exist. Creating...")
                cur.execute(f'CREATE DATABASE "{target_db}"')
            else:
                print(f"[AUTO-DB] Database '{target_db}' already exists.")
    except Exception as e:
        print(f"[AUTO-DB] Error creating DB '{target_db}': {e}")
    finally:
        if 'admin_conn' in locals():
            admin_conn.close()

def connect_import_db(creds):
    """
    1) Using the admin credentials, create the IMPORT_DB if it doesn't exist.
    2) Connect to IMPORT_DB using the normal import user credentials.
    3) Return that connection.
    """
    # Step 1: Get admin credentials
    admin_host = creds["IMPORT_DB_HOST"]
    admin_user = creds["IMPORT_DB_ADMIN_USER"]  # admin user that can create DB
    admin_pass = creds["IMPORT_DB_ADMIN_PASS"]
    target_db_name = creds["IMPORT_DB"]  # the DB name we want to create
    # create the DB
    create_database_if_not_exists(admin_host, admin_user, admin_pass, target_db_name)

    # Step 2: Connect to the newly-created (or existing) DB with normal import user
    import_user = creds["IMPORT_DB_USER"]
    import_pass = creds["IMPORT_DB_PASSWORD"]
    try:
        conn = psycopg2.connect(
            host=admin_host,
            dbname=target_db_name,
            user=import_user,
            password=import_pass
        )
        return conn
    except Exception as e:
        print(f"[IMPORT] Error connecting to import_db '{target_db_name}' as user '{import_user}': {e}")
        return None

def sftp_list_files(sftp, remote_path, file_filter=None):
    # (unchanged)
    all_files = []
    def recursive_list(path):
        try:
            for entry in sftp.listdir_attr(path):
                filename = entry.filename if entry.filename else ""
                full_path = os.path.join(path, filename).replace("\\", "/")
                if stat.S_ISDIR(entry.st_mode):
                    recursive_list(full_path)
                elif file_filter is None or (isinstance(filename, str) and file_filter in filename):
                    all_files.append(full_path)
        except Exception as e:
            print(f"Error accessing {path}: {e}")
    recursive_list(remote_path)
    return all_files

def normalize_remote_path(path):
    return path.replace("\\", "/")

def read_remote_file(sftp, remote_file_path):
    remote_file_path = normalize_remote_path(remote_file_path)
    try:
        with sftp.open(remote_file_path, 'rb') as remote_file:
            if remote_file_path.endswith('.gz'):
                with gzip.GzipFile(fileobj=remote_file) as gz_file:
                    content = gz_file.read()
            else:
                content = remote_file.read()
        return content.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Failed to read remote file {remote_file_path}: {e}")
        return None

def compute_file_hash(file_content):
    """Compute a hash for the file content."""
    return hashlib.md5(file_content.encode('utf-8')).hexdigest()

def is_file_imported(connection, file_hash):
    """Check if file_hash is in 'imported_files' table of this import DB connection."""
    if not connection:
        print("[IMPORT] No connection to import_db. Assuming not imported.")
        return False
    cursor = connection.cursor()
    check_query = "SELECT COUNT(*) FROM imported_files WHERE file_hash = %s"
    cursor.execute(check_query, (file_hash,))
    count = cursor.fetchone()[0]
    cursor.close()
    return (count > 0)

def mark_file_as_imported(connection, file_name, file_hash):
    """Insert file_name & file_hash into 'imported_files' table. Uses ON CONFLICT (file_hash) DO NOTHING."""
    if not connection:
        print("[IMPORT] No connection to import_db. Can't mark file as imported.")
        return
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO imported_files (file_name, file_hash)
    VALUES (%s, %s)
    ON CONFLICT (file_hash) DO NOTHING
    """
    cursor.execute(insert_query, (file_name, file_hash))
    connection.commit()
    cursor.close()

def create_imported_files_table(connection):
    """Create 'imported_files' table in import_db if not exists."""
    if not connection:
        print("[IMPORT] No connection to import_db. Can't create table.")
        return
    cursor = connection.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS imported_files (
        id SERIAL PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_hash TEXT NOT NULL UNIQUE,
        import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    cursor.execute(create_table_query)
    connection.commit()
    cursor.close()

def get_all_user_tables(pg_conn):
    """
    Returns only tables where the name is 'R' followed by exactly 10 digits.
    """
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
      AND tablename ~ '^R[0-9]{10}$'  -- Regex to match "R" + 10 digits
    """
    with pg_conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]