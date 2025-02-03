import os
import stat


def sftp_list_files(sftp, remote_path, file_filter=None):
    all_files = []
    def recursive_list(path):
        try:
            for entry in sftp.listdir_attr(path):
                filename = entry.filename if entry.filename else ""  # Ensure filename is a string
                full_path = os.path.join(path, filename).replace("\\", "/")
                if stat.S_ISDIR(entry.st_mode):
                    recursive_list(full_path)
                elif file_filter is None or (isinstance(filename, str) and file_filter in filename):
                    all_files.append(full_path)
        except Exception as e:
            print(f"Error accessing {path}: {e}")

    recursive_list(remote_path)
    return all_files


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
    """Check if a file with the given hash has already been imported."""
    cursor = connection.cursor()
    check_query = """
    SELECT COUNT(*) FROM imported_files WHERE file_hash = %s
    """
    cursor.execute(check_query, (file_hash,))
    count = cursor.fetchone()[0]
    cursor.close()
    return count > 0

def mark_file_as_imported(connection, file_name, file_hash):
    """Mark a file as imported by inserting its name and hash into the database."""
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO imported_files (file_name, file_hash) VALUES (%s, %s)
    """
    cursor.execute(insert_query, (file_name, file_hash))
    connection.commit()
    cursor.close()


def create_imported_files_table(connection):
    """Create a table to track imported files."""
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
