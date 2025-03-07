#!/usr/bin/env python3
import os
import json
import psycopg2
import pandas as pd
import numpy as np
import plotly.express as px
import argparse
from datetime import datetime, timezone


def load_db_credentials():
    """Load PostgreSQL credentials from credentials.txt"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    credentials_file = os.path.join(base_dir, "credentials.txt")
    try:
        with open(credentials_file, "r") as f:
            creds = json.load(f)
            return {
                "DB_HOST": creds["DB_HOST"],
                "DB_NAME": creds["DB_NAME"],
                "DB_USER": creds["DB_USER"],
                "DB_PASSWORD": creds["DB_PASSWORD"]
            }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading credentials: {e}")
        return None


DB_CREDENTIALS = load_db_credentials()


def pg_connect():
    """Establishes a database connection."""
    if not DB_CREDENTIALS:
        return None
    try:
        conn = psycopg2.connect(
            host=DB_CREDENTIALS["DB_HOST"],
            database=DB_CREDENTIALS["DB_NAME"],
            user=DB_CREDENTIALS["DB_USER"],
            password=DB_CREDENTIALS["DB_PASSWORD"]
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None


def fetch_anomalies(start_date=None, end_date=None):
    """
    Fetch all data from the 'anomalies' table with an optional date filter.
    Replaces null significance_score with 0, and ensures an identifier column 'rxid'.
    """
    conn = pg_connect()
    if not conn:
        print("❌ No database connection available.")
        return pd.DataFrame()

    try:
        base_query = """
        SELECT "timestamp", 
               COALESCE(rxid, "RxID") AS rxid,
               COALESCE(significance_score, 0) AS significance_score,
               autoenc_is_anomaly, 
               lstm_is_anomaly, 
               sgs_duration_ms,
               event_type, 
               message
        FROM public.anomalies
        WHERE (autoenc_is_anomaly = TRUE
          OR lstm_is_anomaly = TRUE
          OR COALESCE(NULLIF(customer_marked_flag, ''), 'false')::boolean = TRUE
          OR COALESCE(NULLIF(important_investigate::text, ''), 'false')::boolean = TRUE)
        """
        if start_date is not None:
            base_query += " AND \"timestamp\" >= '{}'".format(start_date)
        if end_date is not None:
            base_query += " AND \"timestamp\" <= '{}'".format(end_date)
        base_query += " ORDER BY \"timestamp\" DESC"

        df = pd.read_sql(base_query, conn)
        return df
    except Exception as e:
        print(f"❌ Error fetching anomalies: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def normalize_timestamps(series: pd.Series) -> pd.Series:
    """
    Attempts to parse timestamps with flexible heuristics.
    1) Uses pandas to_datetime with errors='coerce'.
    2) For any NaT values, replaces them with the median of valid timestamps.
    """
    parsed = pd.to_datetime(series, errors='coerce', utc=True)
    valid_times = parsed.dropna()
    if valid_times.empty:
        return parsed
    median_time = valid_times.median()
    parsed = parsed.fillna(median_time)
    return parsed


def process_data(df):
    """Prepares and formats data for plotting."""
    if df.empty:
        print("⚠️ No data found.")
        return None

    # Normalize timestamps
    df["timestamp"] = normalize_timestamps(df["timestamp"])
    # Create a version of timestamp that ignores the year (e.g., set year to 2000)
    df["timestamp_no_year"] = df["timestamp"].apply(lambda dt: dt.replace(year=2000))
    # Convert timestamps to numeric (Unix timestamp in seconds) based on the normalized date
    df["timestamp_numeric"] = df["timestamp_no_year"].apply(lambda x: x.timestamp())

    # Ensure significance_score is numeric; if null, set to 0
    df["significance_score"] = pd.to_numeric(df["significance_score"], errors='coerce').fillna(0)

    # Assign colors based on anomaly flags and signal protector keywords.
    # If event_type contains "ipll", "ipvod", or "llot" (case-insensitive), mark as yellow.
    yellow_condition = df["event_type"].str.contains(r"ipll|ipvod|llot", case=False, na=False)
    df["color"] = np.where(
        yellow_condition, "yellow",
        np.where(
            (df["autoenc_is_anomaly"] == True) & (df["lstm_is_anomaly"] == True), "black",
            np.where(df["autoenc_is_anomaly"] == True, "red",
                     np.where(df["lstm_is_anomaly"] == True, "orange", "blue"))
        )
    )

    # Normalize event duration for size scaling; if missing, default to 1.
    df["size"] = pd.to_numeric(df["sgs_duration_ms"], errors='coerce').fillna(1)
    if df["size"].max() > 0:
        df["size"] = np.clip(df["size"] / df["size"].max() * 20, 5, 30)
    else:
        df["size"] = 5

    return df


def plot_3d_anomalies(df):
    """Generates a 3D scatter plot using Plotly."""
    fig = px.scatter_3d(
        df,
        x="timestamp_numeric",
        y="rxid",
        z="significance_score",
        color="color",
        size="size",
        hover_data=["timestamp", "event_type", "message"],
        title="3D Anomaly Correlation (Ignoring Year) - RxID vs Time vs Significance"
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Time (Unix Timestamp, Year Ignored)",
            yaxis_title="Receiver ID",
            zaxis_title="Anomaly Significance",
        ),
        legend_title="Anomaly Type",
        template="plotly_dark"
    )
    fig.show()


def main():
    parser = argparse.ArgumentParser(description="3D Anomaly Plot with Date and Receiver ID Filters")
    parser.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--rxids", type=str, help="Comma-separated list of receiver IDs to filter")
    args = parser.parse_args()

    rxid_list = None
    if args.rxids:
        rxid_list = [r.strip() for r in args.rxids.split(",")]

    df = fetch_anomalies(args.start_date, args.end_date)
    if rxid_list:
        # Filter the DataFrame to include only the specified receiver IDs
        df = df[df["rxid"].isin(rxid_list)]

    df = process_data(df)
    if df is not None:
        print(f"[INFO] Plotting {len(df)} anomaly records.")
        plot_3d_anomalies(df)


if __name__ == "__main__":
    main()
