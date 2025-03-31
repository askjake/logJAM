#!/usr/bin/env python3
"""
video_audio_analyzer.py

Analyzes logs for VAR-style events:
  - CHANGE_CONTENT lines
  - Tally events such as BlackScreen, video freeze, display drop, and trick modes.
Generates a JSON summary per table, stores these summaries in a chart table
in the DB "3090" (for Superset queries), and displays the summary via a Flask web server.
"""

import argparse
import json
import logging
import os
import psycopg2
from psycopg2.extras import DictCursor
from flask import Flask, jsonify

app = Flask(__name__)
analysis_results = {}  # Global in-memory summary keyed by table_name


########################################
# Database Connection Helpers
########################################
def connect_to_db():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        print("[ERROR] credentials.txt not found at", creds_path)
        return None
    try:
        with open(creds_path, "r") as f:
            creds = json.load(f)
        conn = psycopg2.connect(
            host=creds["DB_HOST"],
            database=creds["DB_NAME"],
            user=creds["DB_USER"],
            password=creds["DB_PASSWORD"]
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Logs DB connection failed: {e}")
        return None


def connect_to_chart_db():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        print("[ERROR] credentials.txt not found at", creds_path)
        return None
    try:
        with open(creds_path, "r") as f:
            creds = json.load(f)
        conn = psycopg2.connect(
            host=creds["3090_HOST"],
            database=creds["3090_db"],
            user=creds["3090_USER"],
            password=creds["3090_PASSWORD"]
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Chart DB connection failed: {e}")
        return None


########################################
# Log Fetching & Analysis Functions
########################################
def fetch_all_r_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname = 'public'
              AND tablename ~ '^R[0-9]{10}$'
        """)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def fetch_logs(conn, table_name, start_time=None, end_time=None):
    query = f'SELECT timestamp, data, event_type FROM "{table_name}"'
    filters = []
    if start_time:
        filters.append(f""" "timestamp" >= '{start_time}' """)
    if end_time:
        filters.append(f""" "timestamp" <= '{end_time}' """)
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY timestamp ASC"
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query)
        logs = cur.fetchall()
    return logs


def extract_svc_name(line_data):
    import re
    match = re.search(r"svc_name\s*=\s*([\w\-]+)", line_data, re.IGNORECASE)
    return match.group(1) if match else None


def extract_video_resolution(line_data):
    import re
    match = re.search(r"video resolution[:\s]+(\d{3,4}x\d{3,4})", line_data, re.IGNORECASE)
    return match.group(1) if match else None


def parse_display_drop_value(line_data):
    import re
    match = re.search(r"Display Drop Detected\s*=\s*(\d+)", line_data)
    return int(match.group(1)) if match else 0


def analyze_var_logs(logs):
    """
    Process logs to break them into sessions based on CHANGE_CONTENT lines.
    If no CHANGE_CONTENT line is found, the first log row starts a session.
    Within each session, events for BlackScreen, video freeze, display drop, etc.
    are tallied.
    """
    sessions = []
    current_session = None

    def finalize_session():
        nonlocal current_session
        if current_session is not None:
            # Even if no CHANGE_CONTENT marker was present, ensure a session is recorded.
            sessions.append(current_session)

    for row in logs:
        line_data = (row["data"] or "").strip()
        lower_line = line_data.lower()

        # If a CHANGE_CONTENT line is encountered...
        if "change_content" in lower_line:
            if current_session is not None:
                finalize_session()
            # Start a new session using the current row
            current_session = {
                "change_content_timestamp": row["timestamp"],
                "svc_name": extract_svc_name(line_data),
                "video_resolution": extract_video_resolution(line_data),
                "alerts": {
                    "BlackScreen": 0,
                    "VideoFreeze": 0,
                    "DisplayDrop_0_99": 0,
                    "DisplayDrop_100_299": 0,
                    "DisplayDrop_300_499": 0,
                    "DisplayDrop_500_699": 0,
                    "DisplayDrop_700_plus": 0
                },
                "other_events": []
            }
            continue

        # If no session is active, start one with the first row encountered.
        if current_session is None:
            current_session = {
                "change_content_timestamp": row["timestamp"],
                "svc_name": None,
                "video_resolution": None,
                "alerts": {
                    "BlackScreen": 0,
                    "VideoFreeze": 0,
                    "DisplayDrop_0_99": 0,
                    "DisplayDrop_100_299": 0,
                    "DisplayDrop_300_499": 0,
                    "DisplayDrop_500_699": 0,
                    "DisplayDrop_700_plus": 0
                },
                "other_events": []
            }

        # Tally events in the current session.
        if "blackscreen" in lower_line:
            current_session["alerts"]["BlackScreen"] += 1
        if "video freeze" in lower_line:
            current_session["alerts"]["VideoFreeze"] += 1
        if "display drop detected" in lower_line:
            try:
                x = parse_display_drop_value(line_data)
            except Exception:
                x = 0
            if x < 100:
                current_session["alerts"]["DisplayDrop_0_99"] += 1
            elif x < 300:
                current_session["alerts"]["DisplayDrop_100_299"] += 1
            elif x < 500:
                current_session["alerts"]["DisplayDrop_300_499"] += 1
            elif x < 700:
                current_session["alerts"]["DisplayDrop_500_699"] += 1
            else:
                current_session["alerts"]["DisplayDrop_700_plus"] += 1
        if "enterstandby" in lower_line or "day dream" in lower_line:
            current_session["other_events"].append({
                "timestamp": row["timestamp"],
                "description": line_data
            })

    if current_session is not None:
        finalize_session()
    return sessions


########################################
# 7) Store Analysis Results in Chart DB
########################################
def create_analysis_table(chart_conn):
    with chart_conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'video_audio_analysis'
              AND column_name = 'table_name'
        """)
        row = cur.fetchone()
        if row:
            if row[1].lower() != 'text':
                logging.info("Altering column 'table_name' to TEXT...")
                cur.execute("ALTER TABLE video_audio_analysis ALTER COLUMN table_name TYPE TEXT USING table_name::text")
                chart_conn.commit()
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_audio_analysis (
                    id SERIAL PRIMARY KEY,
                    table_name TEXT,
                    analysis_json JSONB,
                    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            chart_conn.commit()


def store_analysis_results(results):
    chart_conn = connect_to_chart_db()
    if not chart_conn:
        print("[ERROR] Could not connect to chart DB (3090).")
        return
    create_analysis_table(chart_conn)
    upsert_sql = """
    INSERT INTO video_audio_analysis (table_name, analysis_json)
    VALUES (%s, %s::jsonb)
    ON CONFLICT (table_name)
    DO UPDATE SET analysis_json = EXCLUDED.analysis_json,
                  analysis_time = CURRENT_TIMESTAMP;
    """
    with chart_conn.cursor() as cur:
        for tbl, summary in results.items():
            cur.execute(upsert_sql, (tbl, json.dumps(summary, indent=2, default=str)))
    chart_conn.commit()
    chart_conn.close()
    print("[INFO] Analysis results stored in chart DB (video_audio_analysis table).")


########################################
# Flask Endpoints for Display
########################################
@app.route("/")
def index():
    chart_conn = connect_to_chart_db()
    if not chart_conn:
        return "Error connecting to chart DB.", 500
    with chart_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            "SELECT table_name, analysis_json, analysis_time FROM video_audio_analysis ORDER BY analysis_time DESC")
        rows = cur.fetchall()
    chart_conn.close()
    return """
    <html>
      <head><title>Video Audio Analysis Results</title></head>
      <body>
        <h1>Video/Audio Analysis Results</h1>
        <pre>{}</pre>
      </body>
    </html>
    """.format(json.dumps(rows, indent=2, default=str))


########################################
# MAIN
########################################
def main():
    parser = argparse.ArgumentParser(description="Analyze logs for VAR events: CHANGE_CONTENT, BlackScreen, etc.")
    parser.add_argument("--table_name", type=str, default=None,
                        help="R+10 table to analyze, or 'all' for all matching tables.")
    parser.add_argument("--start_time", type=str, default=None,
                        help="Start timestamp for logs filter (ISO format).")
    parser.add_argument("--end_time", type=str, default=None,
                        help="End timestamp for logs filter (ISO format).")
    args = parser.parse_args()

    conn = connect_to_db()
    if not conn:
        print("[ERROR] Could not connect to logs DB.")
        return

    table_names = []
    if not args.table_name or args.table_name.lower() == 'all':
        table_names = fetch_all_r_tables(conn)
    else:
        table_names = [args.table_name]

    for tbl in table_names:
        print(f"[INFO] Processing table: {tbl}")
        logs = fetch_logs(conn, tbl, start_time=args.start_time, end_time=args.end_time)
        if not logs:
            print(f"[WARN] No logs found in {tbl} for the specified timeframe.")
            continue
        session_summaries = analyze_var_logs(logs)
        print(f"[DEBUG] Summary for table {tbl}: {json.dumps(session_summaries, indent=2, default=str)}")
        analysis_results[tbl] = session_summaries

    conn.close()

    store_analysis_results(analysis_results)
    # To test via Flask, you can uncomment the following:
    # app.run(debug=True, port=5001)


if __name__ == "__main__":
    main()
