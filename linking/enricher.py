import os
import re
import json
import argparse
import psycopg2
from psycopg2.extras import DictCursor
from psycopg2 import sql

def data(log_data):
    """Extracts structured information from log lines."""
    data_str = log_data.get("data", "")

    # Extract Type
    type_match = re.search(r'Type:<(\d+):([\w_]+)>', data_str)
    if type_match:
        log_data["msg_type_code"] = int(type_match.group(1))
        log_data["msg_type_name"] = type_match.group(2)

    # Extract Class
    class_match = re.search(r'Class:<(\d+):([\w_]+)>', data_str)
    if class_match:
        log_data["msg_class_code"] = int(class_match.group(1))
        log_data["msg_class_name"] = class_match.group(2)

    # Extract key parameters (stbh, cid, sid, sessMode, etc.)
    field_patterns = {
        "stbh": r"stbh:<([^>]+)>",
        "cid": r"cid:<(\d+)>",
        "sid": r"sid:<(\d+)>",
        "sessMode": r"sessMode:<(\d+)>"
    }

    for field, pattern in field_patterns.items():
        match = re.search(pattern, data_str)
        if match:
            log_data[field] = match.group(1)

    # Label event types for easy classification
    known_events = {
        "ES_STB_MSG_TYPE_CHAN_LIST_EVENT_TRANSITION": "CHAN_LIST_EVENT_TRANSITION",
        "ES_STB_MSG_TYPE_EPG_CUR_PROGRAM_ALL": "EPG_CUR_PROGRAM_ALL",
        "ES_STB_MSG_TYPE_RCA_GET_GET_TUNE_TO_SERVICE": "RCA_GET_TUNE_TO_SERVICE"
    }
    if log_data.get("msg_type_name") in known_events:
        log_data["event_type"] = known_events[log_data["msg_type_name"]]

    return log_data

def parse_schema_and_table(table_name: str):
    """
    Splits a fully qualified table name (like 'public.bad') into (schema, table).
    If there's no dot, default to (public, table_name).
    Also handles any leading/trailing spaces or quotes.
    """
    table_name = table_name.strip().strip('"')

    if not table_name:
        return None, None

    if '.' in table_name:
        parts = table_name.split('.', 1)
        schema_part = parts[0].strip().strip('"') or 'public'
        table_part = parts[1].strip().strip('"')
    else:
        schema_part = 'public'
        table_part = table_name

    return schema_part, table_part

def get_all_user_tables(pg_conn):
    """
    Returns a list of user-defined tables in the 'public' schema, ignoring system tables.
    """
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
      AND tablename NOT LIKE 'pg_%'
      AND tablename NOT LIKE 'sql_%';
    """
    with pg_conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]

def table_has_columns(pg_conn, schema_name, table_name, columns):
    """
    Returns True if all columns in `columns` exist in schema_name.table_name,
    otherwise False.
    """
    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
        """, (schema_name, table_name))
        existing = {row[0].lower() for row in cur.fetchall()}

    for col in columns:
        if col.lower() not in existing:
            return False
    return True

def ensure_columns_exist(pg_conn, schema_name, table_name, columns):
    """
    Dynamically adds columns to schema_name.table_name if they do not already exist.
    We assume each column is TEXT for simplicity. 
    Uses IF NOT EXISTS to avoid "duplicate column" errors.
    """
    with pg_conn.cursor() as cur:
        for col in columns:
            alter_query = sql.SQL('ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS {} TEXT').format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                sql.Identifier(col)
            )
            cur.execute(alter_query)
            # If column didn't exist, it will be added quietly, else ignored
    pg_conn.commit()

def enrich_table_logs(pg_conn, full_table_name, limit=None):
    """
    Fetches rows from 'full_table_name', calls enrich_log_data on each row,
    and updates the table with any new fields extracted.
    Optionally limit the number of rows processed for demonstration or performance.

    We expect the table to have at least:
      - id
      - data
    If not found, we skip.
    """
    schema_part, table_part = parse_schema_and_table(full_table_name)
    if not table_part:
        print(f"[ENRICHER] Invalid table name '{full_table_name}'. Skipping.")
        return

    # 1) Check if table has the minimum columns we need: id, data
    if not table_has_columns(pg_conn, schema_part, table_part, ["id", "data"]):
        print(f"[ENRICHER] Table '{schema_part}.{table_part}' missing 'id' or 'data' column. Skipping.")
        return

    # 2) Ensure target columns exist (IF NOT EXISTS)
    required_columns = [
        "msg_type_code", "msg_type_name", 
        "msg_class_code", "msg_class_name",
        "stbh", "cid", "sid", "sessMode",
        "event_type"
    ]
    ensure_columns_exist(pg_conn, schema_part, table_part, required_columns)

    # 3) Fetch rows
    fetch_sql = sql.SQL("SELECT id, data FROM {}.{} ORDER BY id ASC").format(
        sql.Identifier(schema_part),
        sql.Identifier(table_part)
    )
    if limit:
        fetch_sql = sql.SQL("{} LIMIT {}").format(fetch_sql, sql.Literal(limit))

    rows = []
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(fetch_sql)
        rows = cur.fetchall()

    if not rows:
        print(f"[ENRICHER] No rows found in table '{schema_part}.{table_part}'. Skipping.")
        return

    # 4) For each row, parse data, build an UPDATE
    with pg_conn.cursor() as cur:
        for row in rows:
            row_id = row["id"]
            data_str = row["data"] or ""
            log_data = {"data": data_str}
            enriched = enrich_log_data(log_data)

            update_fields = {}
            for col in required_columns:
                if col in enriched and enriched[col] is not None:
                    update_fields[col] = str(enriched[col])

            if update_fields:
                set_clauses = []
                params = []
                for k, v in update_fields.items():
                    set_clauses.append(sql.SQL('{} = %s').format(sql.Identifier(k)))
                    params.append(v)
                params.append(row_id)

                update_query = sql.SQL("UPDATE {}.{} SET ").format(
                    sql.Identifier(schema_part),
                    sql.Identifier(table_part)
                )
                update_query += sql.SQL(", ").join(set_clauses)
                update_query += sql.SQL(" WHERE id = %s")

                cur.execute(update_query, params)

    pg_conn.commit()
    print(f"[ENRICHER] Enriched {len(rows)} row(s) in '{schema_part}.{table_part}'.")

def main():
    parser = argparse.ArgumentParser(description="Log Enricher: Extract structured fields and store them in DB.")
    parser.add_argument("--table_name", type=str,
                        help="If provided, only enrich this table (optionally specify schema.table). Otherwise, enrich all user tables.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on the number of rows to process per table.")
    args = parser.parse_args()

    # Step 1: credentials.txt in base dir
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    credentials_file = os.path.join(base_dir, "credentials.txt")

    if not os.path.exists(credentials_file):
        print(f"❌ credentials.txt not found at: {credentials_file}")
        return

    # Step 2: Load credentials
    try:
        with open(credentials_file, "r") as f:
            creds = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON in {credentials_file}: {e}")
        return

    db_host = creds.get("DB_HOST")
    db_name = creds.get("DB_NAME")
    db_user = creds.get("DB_USER")
    db_pass = creds.get("DB_PASSWORD")

    # Step 3: Connect to PostgreSQL
    try:
        pg_conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_pass
        )
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL at {db_host}: {e}")
        return 

    # Step 4: Enrich either one table or all user-defined tables
    if args.table_name:
        print(f"[ENRICHER] Enriching table '{args.table_name}' only...")
        enrich_table_logs(pg_conn, args.table_name, limit=args.limit)
    else:
        with pg_conn.cursor() as cur:
            all_tables = get_all_user_tables(pg_conn)
        print(f"[ENRICHER] Found {len(all_tables)} user-defined tables in '{db_name}'.")
        for t in all_tables:
            print(f"\n[ENRICHER] Enriching table '{t}' ...")
            enrich_table_logs(pg_conn, t, limit=args.limit)

    pg_conn.close()
    print("[ENRICHER] Completed enrichment. Connection closed.")

if __name__ == "__main__":
    main()
