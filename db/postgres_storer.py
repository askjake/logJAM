# db/postgres_storer.py

import psycopg2
from psycopg2.extras import execute_values

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
    cursor = connection.cursor()
    # Changed unique constraint to data_hash only
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


    cursor = connection.cursor()

    def get_existing_columns():
        cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns 
            WHERE table_name = %s
        """, (table_name.lower(),))
        return {row[0].lower(): row[0] for row in cursor.fetchall()}  # Map lowercase to actual column name

    def ensure_column_types(existing_columns):
        # If needed, ensure 'data' is TEXT:
        # We'll assume it's already TEXT from initial creation
        pass

    def add_missing_columns(existing_columns, new_columns):
        # existing_columns keys are lowercase for comparison
        # new_columns are actual keys from logs
        for column in new_columns:
            col_lower = column.lower()
            if col_lower not in existing_columns and column != "id":
                # Add column with exact case from 'column'
                alter_query = f"""ALTER TABLE "{table_name}" ADD COLUMN "{column}" TEXT"""
                cursor.execute(alter_query)
                print(f"Added column '{column}' to table '{table_name}'")
        connection.commit() 

    def insert_logs():
        # Ensure message column exists
        columns = list(logs[0].keys())
        if "message" not in columns:
            for log in logs:
                log["message"] = "N/A"
        if "data_hash" not in columns:
            for log in logs:
                directory_file = log.get("directory_file")
                ts = log.get("timestamp")
                d = log.get("data")
                log["data_hash"] = compute_data_hash(directory_file, ts, d)

        columns = list(logs[0].keys())
        values = [tuple(log.get(col, None) for col in columns) for log in logs]

        insert_query = f"""
            INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
            VALUES %s
            ON CONFLICT (data_hash) DO NOTHING
        """

        execute_values(cursor, insert_query, values)
        connection.commit()

        rows_attempted = len(values)
        rows_inserted = cursor.rowcount
        duplicates = rows_attempted - rows_inserted
        duplicate_percentage = (duplicates / rows_attempted) * 100 if rows_attempted > 0 else 0
        print(f"Duplicates: {duplicate_percentage:6.2f}%  | File: {original_log_file_name:<40} | Rows Added: {rows_inserted:<8}| To Table: {table_name}")

    try:
        existing_columns = get_existing_columns()
        # Gather all keys from all logs
        new_columns = {key for log in logs for key in log.keys()}
        add_missing_columns(existing_columns, new_columns)
        ensure_column_types(existing_columns)
        insert_logs()
    except Exception as e:
        print(f"Error storing logs in table '{table_name}': {e}")
        connection.rollback()
    finally:
        cursor.close()


def store_parsed_logs(connection, table_name, logs, original_log_file_name):
    cursor = connection.cursor()

    def get_existing_columns():
        cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns 
            WHERE table_name = %s
        """, (table_name.lower(),))
        return {row[0].lower(): row[0] for row in cursor.fetchall()}  # Map lowercase to actual column name

    def ensure_column_types(existing_columns):
        # If needed, ensure 'data' is TEXT:
        # We'll assume it's already TEXT from initial creation
        pass

    def add_missing_columns(existing_columns, new_columns):
        # existing_columns keys are lowercase for comparison
        # new_columns are actual keys from logs
        for column in new_columns:
            col_lower = column.lower()
            if col_lower not in existing_columns and column != "id":
                # Add column with exact case from 'column'
                alter_query = f"""ALTER TABLE "{table_name}" ADD COLUMN "{column}" TEXT"""
                cursor.execute(alter_query)
                print(f"Added column '{column}' to table '{table_name}'")
        connection.commit() 

    def insert_logs():
        # Ensure message column exists
        columns = list(logs[0].keys())
        if "message" not in columns:
            for log in logs:
                log["message"] = "N/A"
        if "data_hash" not in columns:
            for log in logs:
                directory_file = log.get("directory_file")
                ts = log.get("timestamp")
                d = log.get("data")
                log["data_hash"] = compute_data_hash(directory_file, ts, d)

        columns = list(logs[0].keys())
        values = [tuple(log.get(col, None) for col in columns) for log in logs]

        insert_query = f"""
            INSERT INTO "{table_name}" ({', '.join(f'"{col}"' for col in columns)})
            VALUES %s
            ON CONFLICT (data_hash) DO NOTHING
        """

        execute_values(cursor, insert_query, values)
        connection.commit()

        rows_attempted = len(values)
        rows_inserted = cursor.rowcount
        duplicates = rows_attempted - rows_inserted
        duplicate_percentage = (duplicates / rows_attempted) * 100 if rows_attempted > 0 else 0
        print(f"Duplicates: {duplicate_percentage:6.2f}%  | File: {original_log_file_name:<40} | Rows Added: {rows_inserted:<8}| To Table: {table_name}")

    try:
        existing_columns = get_existing_columns()
        # Gather all keys from all logs
        new_columns = {key for log in logs for key in log.keys()}
        add_missing_columns(existing_columns, new_columns)
        ensure_column_types(existing_columns)
        insert_logs()
    except Exception as e:
        print(f"Error storing logs in table '{table_name}': {e}")
        connection.rollback()
    finally:
        cursor.close()