import os
import hashlib
import psycopg2
from datetime import datetime

# Database connection details (hardcoded)
DB_HOST = "10.74.139.250"
DB_NAME = "versioncontroldb"
DB_USER = "dishiptest"
DB_PASS = "Dish1234"
TABLE_NAME = "version_control"

# Exclusion patterns for files and directories
EXCLUSION_PATTERNS = [
    'node_modules', '.git', '.idea', 'apps', '__pycache__', 'vJAM', 
    'vJAMbot', 'vlogJAM', 'lib', 'Lib', '.gitattributes', '.gitignore', 
    'README', '*.pyc', '*.log'
]

# Get the base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_connection():
    """Returns a connection object to the PostgreSQL database."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST
    )


def create_table_if_not_exists(conn):
    """Creates the version_control table if it does not already exist."""
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
    """Calculates the MD5 checksum for the given file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_version_numbers(conn, file_name, file_md5):
    """
    Retrieves both the "Version in Use" (matching MD5) and the highest available version for a file.
    Returns (in_use_version, highest_version).
    """
    query = f"""
        SELECT version_number, md5 FROM {TABLE_NAME} 
        WHERE file_name = %s 
        ORDER BY version_number DESC;
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (file_name,))
        rows = cursor.fetchall()
    
    if not rows:
        return None, 1  # No version exists, file starts at version 1

    highest_version = rows[0][0]  # Latest version number

    # Find the version that matches the current MD5
    in_use_version = next((v[0] for v in rows if v[1] == file_md5), None)

    return in_use_version, highest_version


def record_version_if_new(conn, file_name, file_md5):
    """
    Checks if the file has a new version, and if so, records it.
    Displays both the in-use version and the highest available version.
    """
    in_use_version, highest_version = get_version_numbers(conn, file_name, file_md5)

    insert_query = f"""
        INSERT INTO {TABLE_NAME} (file_name, md5, version_number) 
        VALUES (%s, %s, %s);
    """

    next_version = highest_version + 1 if highest_version else 1

    # Check if the new MD5 matches any existing version
    if in_use_version:
        status = " Up to date" if in_use_version == highest_version else "⚠️ Using older version"
        print(f" - v.{in_use_version}:{highest_version:<3}{status} {file_name}")
    else:
        # Insert new version
        with conn.cursor() as cursor:
            cursor.execute(insert_query, (file_name, file_md5, next_version))
            conn.commit()

        print(f" => v.{next_version:<2}  + New version {file_name} ")


def should_exclude_dir(dirname):
    """Determines if a directory should be excluded."""
    if dirname.startswith('.'):
        return True
    return any(pattern in dirname for pattern in EXCLUSION_PATTERNS)


def should_exclude_file(filename):
    """Determines if a file should be excluded."""
    return any(pattern in filename for pattern in EXCLUSION_PATTERNS)


def get_all_files(base_dir):
    """Recursively finds all files in base_dir, excluding specified patterns."""
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
        for file in files:
            if not should_exclude_file(file):
                all_files.append(os.path.join(root, file))
    return all_files


def main():
    """Main function to track file versions."""
    try:
        conn = get_connection()
        create_table_if_not_exists(conn)

        print(f"\nTracking files in '{BASE_DIR}' and its subdirectories.\n")
        all_files = get_all_files(BASE_DIR)

        for abs_path in all_files:
            if os.path.exists(abs_path):
                rel_path = os.path.relpath(abs_path, BASE_DIR)
                md5_val = calculate_md5(abs_path)
                record_version_if_new(conn, rel_path, md5_val)
            else:
                print(f"Warning: File '{abs_path}' does not exist.")

        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
