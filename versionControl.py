import os
import hashlib
import json
import psycopg2

# 1. Base directory: the folder where this script is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Credentials and DB info
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.txt")
DB_NAME = "versioncontroldb"
TABLE_NAME = "version_control"

EXCLUSION_PATTERNS = [
    'node_modules',
    '.git',
    '.idea',
    'apps',
    '__pycache__',
    'vJAM',
    'vJAMbot',
    'lib',
    # Add more as needed
]

def read_credentials():
    """
    Reads credentials from credentials.txt (JSON).
    Returns: (db_host, db_name, db_user, db_pass) as strings.
    """
    
    db_host = "10.74.139.250"
    db_name = "chatbotdb"
    db_user = "chatbotuser"
    db_pass = "changeme"

    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(f"Could not find {CREDENTIALS_FILE}")
    
    with open(CREDENTIALS_FILE, "r") as f:
        data = json.load(f)
    
    db_host = data.get("DB_HOST", "10.74.139.250")
    db_name = data.get("DB_NAME", "chatbotdb")
    db_user = data.get("DB_USER", "chatbotuser")
    db_pass = data.get("DB_PASSWORD", "changeme")

    if not (db_host and db_user and db_pass):
        raise ValueError("Database credentials are missing or incomplete.")
    
    return db_host, db_name, db_user, db_pass

def get_connection(db_host, db_name, db_user, db_pass):
    """
    Returns a connection object to the specified database.
    """
    return psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_pass,
        host=db_host
    )

def create_table_if_not_exists(conn):
    """
    Creates the version_control table if it does not already exist.
    """
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        file_name TEXT NOT NULL,
        md5 TEXT NOT NULL,
        version_number INT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    with conn.cursor() as cursor:
        cursor.execute(create_table_query)
    conn.commit()

def calculate_md5(file_path):
    """
    Calculates the MD5 checksum for the given file's content.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_next_version_number(conn, file_name):
    """
    Retrieves the latest version_number for the specified file_name
    and returns the next version number.
    """
    query = f"SELECT version_number FROM {TABLE_NAME} WHERE file_name = %s ORDER BY version_number DESC LIMIT 1;"
    with conn.cursor() as cursor:
        cursor.execute(query, (file_name,))
        row = cursor.fetchone()
    return (row[0] + 1) if row else 1

def record_version_if_new(conn, file_name, file_md5):
    """
    Checks if the (file_name, file_md5) combo is already in the table.
    If not, inserts a new row with the next version number.
    """
    check_query = f"SELECT 1 FROM {TABLE_NAME} WHERE file_name = %s AND md5 = %s;"
    insert_query = f"INSERT INTO {TABLE_NAME} (file_name, md5, version_number) VALUES (%s, %s, %s);"

    with conn.cursor() as cursor:
        cursor.execute(check_query, (file_name, file_md5))
        row = cursor.fetchone()
        if row:
            version_number = get_next_version_number(conn, file_name)
            print(f"Active Version: {row[0]:<3}  Newest Version: {version_number:<3}  {file_name} ")
            


        else:
            version_number = get_next_version_number(conn, file_name)
            cursor.execute(insert_query, (file_name, file_md5, version_number))
            conn.commit()
            print(f"New version recorded: {file_name} => Version {version_number}, MD5: {file_md5}")

def should_exclude_dir(dirname):
    """
    Returns True if the directory should be excluded.
    Excludes any directory that:
      - Starts with a dot (hidden folder)
      - Contains any of the patterns from EXCLUSION_PATTERNS
    """
    if dirname.startswith('.'):
        return True
    for pattern in EXCLUSION_PATTERNS:
        if pattern in dirname:
            return True
    return False

def get_all_files(base_dir):
    """
    Recursively collects all files (with absolute paths) in base_dir
    and its subdirectories, skipping directories that match the exclusions.
    """
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        # Remove directories that should be excluded
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

### === Main Routine ===

def main():
    try:
        db_host, db_name, db_user, db_pass = read_credentials()
        conn = get_connection(db_host, db_name, db_user, db_pass)
        create_table_if_not_exists(conn)
        
        print(f"dbhost: {db_host}")
        
        # Get all files in BASE_DIR while excluding specified directories.
        all_files = get_all_files(BASE_DIR)
        print(f"Tracking {len(all_files)} files in '{BASE_DIR}' and its subdirectories.")
        
        for abs_path in all_files:
            if os.path.exists(abs_path):
                # Build a relative path from BASE_DIR for storage
                rel_path = os.path.relpath(abs_path, BASE_DIR)
                # Compute MD5 from the absolute path
                md5_val = calculate_md5(abs_path)
                record_version_if_new(conn, rel_path, md5_val)
            else:
                print(f"Warning: File '{abs_path}' does not exist.")
        
        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()