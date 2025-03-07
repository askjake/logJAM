#!/usr/bin/env python3
import os
import json
import sys
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from concurrent.futures import ThreadPoolExecutor, as_completed
import psycopg2
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
from tkinter import ttk, filedialog, messagebox

def create_sqlalchemy_engine() -> Engine:
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    credentials_file = os.path.join(base_dir, "credentials.txt")
    try:
        with open(credentials_file, "r") as f:
            creds = json.load(f)
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        sys.exit(1)
    db_url = f"postgresql+psycopg2://{creds['DB_USER']}:{creds['DB_PASSWORD']}@{creds['DB_HOST']}/{creds['DB_NAME']}"
    return create_engine(db_url, echo=False)

def ensure_embeddings_table(engine: Engine, embed_table="stb_logs_embeddings"):
    create_table_query = text(f"""
    CREATE TABLE IF NOT EXISTS {embed_table} (
        source_table TEXT NOT NULL,
        log_id INTEGER,
        timestamp TIMESTAMPTZ NOT NULL,
        timestamp_imputed BOOLEAN DEFAULT FALSE,
        embedding JSONB,
        PRIMARY KEY (source_table, log_id)
    );
    """)
    with engine.begin() as conn:
        conn.execute(create_table_query)
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE {embed_table} ADD COLUMN IF NOT EXISTS timestamp_imputed BOOLEAN DEFAULT FALSE;
        """))

def upsert_embeddings_for_table(engine: Engine, df: pd.DataFrame, embed_table="stb_logs_embeddings"):
    if df.empty:
        return 0
    upsert_q = text(f"""
        INSERT INTO {embed_table} (source_table, log_id, timestamp, timestamp_imputed, embedding)
        VALUES (:src, :logid, :ts, :imputed, :emb)
        ON CONFLICT (source_table, log_id) 
        DO UPDATE SET embedding = EXCLUDED.embedding, 
                      timestamp = EXCLUDED.timestamp,
                      timestamp_imputed = EXCLUDED.timestamp_imputed;
    """)
    count = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(upsert_q, {
                "src": row["source_table"],
                "logid": int(row["id"]),
                "ts": row["timestamp"],
                "imputed": row["timestamp_imputed"],
                "emb": json.dumps(row["embedding"])
            })
            count += 1
    return count

def process_table(table: str, engine: Engine) -> dict:
    result = {"table": table, "processed": 0, "updated": False}
    try:
        with engine.connect() as conn:
            source_query = text(f'SELECT MAX("timestamp"), COUNT(*) FROM "{table}";')
            source_res = conn.execute(source_query).fetchone()
            source_max = source_res[0] if source_res and source_res[0] else None
            embed_query = text("""SELECT MAX(timestamp), COUNT(*) FROM stb_logs_embeddings WHERE source_table = :t;""")
            embed_res = conn.execute(embed_query, {"t": table}).fetchone()
            embed_max = embed_res[0] if embed_res else None
        process_flag = (embed_max is None) or (source_max and source_max > embed_max)
        if not process_flag:
            print(f"[INFO] Table {table} is up-to-date. Skipping.")
            return result

        query = f"""
            SELECT id, "timestamp", message, '{table}' AS source_table
            FROM "{table}"
            ORDER BY "timestamp" ASC
            LIMIT 500000;
        """
        df = pd.read_sql(query, engine)
        if df.empty:
            print(f"[WARN] Table {table} is empty.")
            return result
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
        df = embed_logs(df)  # defined below
        upsert_count = upsert_embeddings_for_table(engine, df)
        print(f"[INFO] Processed table {table}: {upsert_count} embeddings upserted.")
        result["processed"] = upsert_count
        result["updated"] = True
    except Exception as e:
        print(f"[ERROR] Failed processing table {table}: {e}")
    return result

def fetch_all_embeddings(engine, start_date=None, end_date=None) -> pd.DataFrame:
    query = text("""
    SELECT log_id, source_table, timestamp, embedding
    FROM stb_logs_embeddings
    WHERE (:start_date IS NULL OR timestamp >= :start_date)
      AND (:end_date IS NULL OR timestamp <= :end_date)
    ORDER BY timestamp ASC
    LIMIT 500000;
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"start_date": start_date, "end_date": end_date})
    df["embedding"] = df["embedding"].apply(lambda x: json.loads(x))
    return df

def plot_embeddings(df: pd.DataFrame):
    from sklearn.decomposition import PCA
    if df.empty:
        print("[WARN] No embeddings to plot.")
        return
    embed_matrix = np.vstack(df["embedding"].values)
    n_components = 3
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embed_matrix)
    for i in range(n_components):
        df[f"pca_{i+1}"] = coords[:, i]
    fig = px.scatter_3d(
        df,
        x="pca_1", y="pca_2", z="pca_3",
        color="source_table",
        hover_data=["log_id", "timestamp"],
        title="STB Log Embeddings (3D PCA)"
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend_title="Source Table",
        template="plotly_dark"
    )
    fig.show()

# New: Modified embed_logs() to use the globally loaded model.
def create_embedding(text: str, model_instance) -> list:
    return model_instance.encode([text])[0].tolist()

def embed_logs(df: pd.DataFrame, model_instance=None) -> pd.DataFrame:
    if df.empty:
        return df
    # Replace NaT with current time
    current_time = datetime.now(timezone.utc)
    df["timestamp"] = df["timestamp"].apply(lambda x: current_time if pd.isna(x) else x)
    df["timestamp_imputed"] = df["timestamp"].apply(lambda x: x == current_time)
    # Use the provided model or default model instance
    if model_instance is None:
        model_instance = SentenceTransformer("all-MiniLM-L6-v2")
    df["embedding"] = df["message"].apply(lambda msg: create_embedding(msg or "", model_instance))
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Process STB log tables concurrently and plot embeddings."
    )
    parser.add_argument("-n", "--num_tables", type=int, default=5,
                        help="Number of tables to process simultaneously.")
    parser.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence Transformer model to use for embedding")

    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    engine = create_sqlalchemy_engine()
    # engine = create_engine("postgresql+psycopg2://user:password@host/dbname")

    if not engine:
        print("[ERR] No engine created. Exiting.")
        return

    ensure_embeddings_table(engine)

    with engine.connect() as conn:
        table_query = text("""
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname = 'public'
              AND tablename ~ '^R[0-9]{10}$';
        """)
        t_result = conn.execute(table_query)
        tables = [row[0] for row in t_result.fetchall()]

    if not tables:
        print("[WARN] No STB log tables found.")
        return

    new_tables = []
    update_tables = []
    with engine.connect() as conn:
        for t in tables:
            embed_q = text("SELECT COUNT(*) FROM stb_logs_embeddings WHERE source_table = :t")
            try:
                e_res = conn.execute(embed_q, {"t": t}).fetchone()
                if e_res[0] == 0:
                    new_tables.append(t)
                else:
                    update_tables.append(t)
            except Exception as e:
                print(f"[WARN] Could not check embeddings for {t}: {e}")

    prioritized_tables = new_tables + update_tables
    print(f"[INFO] Processing {len(prioritized_tables)} tables ({len(new_tables)} new, {len(update_tables)} to update)")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    with ThreadPoolExecutor(max_workers=args.num_tables) as executor:
        future_to_table = {executor.submit(process_table, t, engine): t for t in prioritized_tables}
        for future in as_completed(future_to_table):
            table = future_to_table[future]
            try:
                rdict = future.result()
                results.append(rdict)
            except Exception as exc:
                print(f"[ERROR] Table {table} generated an exception: {exc}")

    for r in results:
        print(f"[SUMMARY] Table {r['table']} => processed: {r['processed']}, updated={r['updated']}")

    embed_df = fetch_all_embeddings(engine)
    # Pass our model instance so embed_logs doesn't reload the default
    embed_df = embed_logs(embed_df, model_instance=model)
    plot_embeddings(embed_df)

if __name__ == "__main__":
    main()
