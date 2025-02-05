import psycopg2
from psycopg2.extras import execute_values
from ingestion.utils import (
    sftp_list_files, read_remote_file, compute_file_hash, 
    is_file_imported, mark_file_as_imported, create_imported_files_table
)

def connect(credentials):
    try:
        return psycopg2.connect(
            host=credentials["DB_HOST"],
            database=credentials["DB_NAME"],
            user=credentials["DB_USER"],
            password=credentials["DB_PASSWORD"]
        )
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def create_table_if_not_exists(connection, table_name):
    """Creates the table if it does not already exist."""
    cursor = connection.cursor()
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        id SERIAL PRIMARY KEY,
        directory_file TEXT NOT NULL,
        category TEXT,
        timestamp TIMESTAMP WITH TIME ZONE,
        file_line TEXT,
        function TEXT,
        data TEXT,
        data_hash TEXT,
        message TEXT,
        UNIQUE (data_hash)
    )
    """
    cursor.execute(create_table_query)
    connection.commit()
    cursor.close()

def get_existing_columns(cursor, table_name):
    """Retrieves existing columns in a table."""
    cursor.execute("""
        SELECT column_name
        FROM information_schema.columns 
        WHERE table_name = %s
    """, (table_name.lower(),))
    return {row[0].lower() for row in cursor.fetchall()}

def add_missing_columns(cursor, table_name, new_columns):
    """Dynamically adds missing columns to the table, ensuring duplicates are not added."""
    existing_columns = get_existing_columns(cursor, table_name)
    for column in new_columns:
        col_lower = column.lower()
        if col_lower not in existing_columns and column != "id":
            try:
                alter_query = f"""ALTER TABLE "{table_name}" ADD COLUMN "{column}" TEXT"""
                cursor.execute(alter_query)
                # Commit after each successful ALTER statement to isolate errors.
                cursor.connection.commit()
                print(f"✅ Added column '{column}' to table '{table_name}'")
            except psycopg2.errors.DuplicateColumn:
                print(f"⚠️ Column '{column}' already exists in '{table_name}', skipping.")
            except Exception as e:
                print(f"❌ Error adding column '{column}' to '{table_name}': {e}")
                # Roll back any partial changes to clear the aborted state before continuing
                cursor.connection.rollback()
        else:
            print(f"⚠️ Column '{column}' already exists in '{table_name}', skipping.")

def store_parsed_logs(connection, table_name, logs, original_log_file_name):
    """Stores parsed logs in the database with error handling and rollback."""
    # Ensure any previous aborted transaction is cleared
    connection.rollback()
    cursor = connection.cursor()

    if not logs:
        print("No logs to insert.")
        cursor.close()
        return

    # Ensure directory_file is present in every log
    for log in logs:
        if "directory_file" not in log or log["directory_file"] is None:
            log["directory_file"] = original_log_file_name

    # Ensure 'message' field exists
    for log in logs:
        if "message" not in log:
            log["message"] = "N/A"

    # Ensure 'data_hash' exists (calculate hash on the data field)
    for log in logs:
        if "data_hash" not in log:
            log["data_hash"] = compute_file_hash(log.get("data", ""))

    try:
        # Get all keys from all log entries.
        new_columns = {key for log in logs for key in log.keys()}
        add_missing_columns(cursor, table_name, new_columns)

        # Recalculate columns after possible additions
        columns = list(logs[0].keys())
        values = [tuple(log.get(col, None) for col in columns) for log in logs]

        insert_query = f"""
            INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
            VALUES %s
            ON CONFLICT (data_hash) DO NOTHING
        """

        try:
            execute_values(cursor, insert_query, values)
            connection.commit()
        except Exception as e:
            print(f"❌ Error inserting logs into '{table_name}': {e}")
            connection.rollback()
            cursor.close()
            return

        rows_attempted = len(values)
        rows_inserted = cursor.rowcount
        duplicates = rows_attempted - rows_inserted
        duplicate_percentage = (duplicates / rows_attempted) * 100 if rows_attempted > 0 else 0
        print(f"✅ Duplicates: {duplicate_percentage:.2f}% | File: {original_log_file_name:<40} | Rows Added: {rows_inserted:<8} | Table: {table_name}")

    except Exception as e:
        print(f"❌ Critical error storing logs in table '{table_name}': {e}")
        connection.rollback()
    finally:
        cursor.close()
