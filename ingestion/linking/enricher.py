#!/usr/bin/env python3
"""
enricher.py

Enhancements:
 1. If parse_data_fields() detects "customer_marked_flag=YES",
    we automatically set event_type="MARKED_LOGS" in that row
    (if no event_type was previously set).
 2. A new function highlight_customer_marked() is called after enrich_table()
    to give a significance boost + mark logs as "important_investigate=TRUE"
    in a specified time window around the marked log time.
 3. The single global model still operates as before, training on all tables combined
    or only the specified table, but we can now incorporate the new "MARKED_LOGS"
    label into the training data for more emphasis.
 4. NEW: For records whose event_type contains any of the keywords 'ipll', 'ipvod', or 'llot'
    (case-insensitive), append ", signal_protector" to the event_type.
"""

import os
import json
import argparse
import psycopg2
from psycopg2 import sql
import subprocess
from psycopg2.extras import DictCursor

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import mutual_info_score
    sklearn_installed = True
except ImportError:
    sklearn_installed = False
    print("[WARN] scikit-learn not installed. ML-based classification won't run automatically.")


###############################################################################
# 1) Utility: Fetch All User Tables
###############################################################################
def get_all_user_tables(pg_conn):
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
      AND tablename ~ '^R[0-9]{10}$'
    """
    with pg_conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]


###############################################################################
# 2) Regex-based Data Extraction
###############################################################################
# ... (existing regex patterns) ...

MARK_LOGS_PATTERN = "ES_STB_MSG_TYPE_VIDEO_DUMP_BRCM_PROC"
DUMP_FILE_PATTERN = "DUMPING FILE "

def parse_data_fields(data_value: str) -> dict:
    """
    Extracts various fields from the 'data' column using regex patterns.
    If we detect strings like "ES_STB_MSG_TYPE_VIDEO_DUMP_BRCM_PROC" or "DUMPING FILE",
    we also set 'customer_marked_flag=YES'.
    """
    extracted = {}
    data_str = str(data_value or "").strip().strip('"')

    # Check for customer-marked patterns
    if MARK_LOGS_PATTERN in data_str or DUMP_FILE_PATTERN in data_str:
        extracted["customer_marked_flag"] = "YES"
        print(f"\r  ⚠️ **** LOGS MARKED ****", end=' ')

    # ... (other regex extraction logic) ...

    return extracted


###############################################################################
# 3) ML Config Table Management
###############################################################################
def ensure_ml_config_table(enrich_conn):
    with enrich_conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ml_config (
                config_key TEXT PRIMARY KEY,
                config_value JSONB
            );
        """)
    enrich_conn.commit()

def load_enricher_config(enrich_conn):
    with enrich_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("SELECT config_value FROM ml_config WHERE config_key = 'enricher_settings'")
        row = cur.fetchone()
        if row:
            return row["config_value"]
    return None

def store_enricher_config(enrich_conn, config):
    with enrich_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ml_config (config_key, config_value)
            VALUES ('enricher_settings', %s)
            ON CONFLICT (config_key)
            DO UPDATE SET config_value = EXCLUDED.config_value
        """, (json.dumps(config),))
    enrich_conn.commit()


###############################################################################
# 4) Table Column Management
###############################################################################
def parse_schema_and_table(table_name: str):
    table_name = table_name.strip().strip('"')
    if '.' in table_name:
        parts = table_name.split('.', 1)
        schema_part = parts[0].strip().strip('"') or 'public'
        table_part = parts[1].strip().strip('"')
    else:
        schema_part = 'public'
        table_part = table_name
    return schema_part, table_part

def get_table_columns(pg_conn, schema_name, table_name):
    q = sql.SQL("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
    """)
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(q, (schema_name, table_name))
        rows = cur.fetchall()
    return {row["column_name"]: row["data_type"] for row in rows}

def add_col_if_missing(cur, schema_name, table_name, col_name, col_type="TEXT"):
    existing = get_table_columns(cur.connection, schema_name, table_name)
    if col_name in existing:
        return
    sql_add = f'ALTER TABLE "{schema_name}"."{table_name}" ADD COLUMN "{col_name}" {col_type}'
    cur.execute(sql_add)

def _add_col(cur, schema_part, table_part, col_name):
    alter_q = sql.SQL('ALTER TABLE {}.{} ADD COLUMN {} TEXT').format(
        sql.Identifier(schema_part),
        sql.Identifier(table_part),
        sql.Identifier(col_name)
    )
    try:
        cur.execute(alter_q)
    except psycopg2.Error:
        pass

def rename_and_migrate_column(cur, schema_part, table_part, old_name, new_name):
    if old_name.lower() == new_name.lower():
        return
    existing = get_table_columns(cur.connection, schema_part, table_part)
    if new_name in existing:
        return
    print(f"[REFORM] Renaming '{old_name}' => '{new_name}' in {schema_part}.{table_part}")
    _add_col(cur, schema_part, table_part, new_name)
    try:
        copy_query = sql.SQL('UPDATE {}.{} SET {} = {} WHERE {} IS NOT NULL').format(
            sql.Identifier(schema_part),
            sql.Identifier(table_part),
            sql.Identifier(new_name),
            sql.Identifier(old_name),
            sql.Identifier(old_name)
        )
        cur.execute(copy_query)
        print(f"  Copied data from '{old_name}' to '{new_name}'.")
    except Exception as e:
        print(f"  ❌ Error copying data from '{old_name}' to '{new_name}': {e}")
        cur.connection.rollback()
        return
    try:
        drop_query = sql.SQL('ALTER TABLE {}.{} DROP COLUMN {}').format(
            sql.Identifier(schema_part),
            sql.Identifier(table_part),
            sql.Identifier(old_name)
        )
        cur.execute(drop_query)
        print(f"  Dropped old column '{old_name}'.")
    except Exception as e:
        print(f"  ⚠️ Could not drop '{old_name}': {e}")
        cur.connection.rollback()

def add_col(cur, schema_part, table_part, col_name):
    existing_cols = get_table_columns(cur.connection, schema_part, table_part)
    if col_name in existing_cols:
        return
    col_type = "BIGINT" if col_name.endswith("_num") else "TEXT"
    print(f"[REFORM] Adding missing column '{col_name}' ({col_type}) in {schema_part}.{table_part}")
    qry = sql.SQL('ALTER TABLE {}.{} ADD COLUMN {} ' + col_type).format(
        sql.Identifier(schema_part),
        sql.Identifier(table_part),
        sql.Identifier(col_name)
    )
    cur.execute(qry)

def refactor_table_for_enrichment(pg_conn, table_name, columns_needed):
    schema_part, table_part = parse_schema_and_table(table_name)
    with pg_conn.cursor() as cur:
        existing_cols = get_table_columns(pg_conn, schema_part, table_part)
        existing_lower_map = {k.lower(): k for k in existing_cols.keys()}
        if "event_type" not in columns_needed:
            columns_needed.append("event_type")
        for c in columns_needed:
            c_lower = c.lower()
            if c_lower in existing_lower_map:
                real_name = existing_lower_map[c_lower]
                if real_name != c:
                    rename_and_migrate_column(cur, schema_part, table_part, real_name, c)
            else:
                add_col(cur, schema_part, table_part, c)
    pg_conn.commit()


###############################################################################
# 5) Detecting "Relevant Columns" from Tables
###############################################################################
def sample_table_rows(pg_conn, schema_part, table_part, sample_size, col_list):
    col_select = ", ".join(f'"{c}"' for c in col_list)
    q = f'''
        SELECT {col_select}
        FROM "{schema_part}"."{table_part}"
        ORDER BY random()
        LIMIT {sample_size}
    '''
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        try:
            cur.execute(q)
            rows = cur.fetchall()
        except Exception as e:
            print(f"[WARN] sampling {schema_part}.{table_part} failed: {e}")
            return []
    return rows

def detect_relevant_columns_table(pg_conn, table_name, sample_size=1000):
    schema_part, table_part = parse_schema_and_table(table_name)
    col_info = get_table_columns(pg_conn, schema_part, table_part)
    text_columns = [c for c, dtype in col_info.items() if 'text' in dtype or 'char' in dtype]
    if 'event_type' not in text_columns:
        text_columns.append('event_type')
    sample_rows = sample_table_rows(pg_conn, schema_part, table_part, sample_size, text_columns)
    if not sample_rows:
        return list(set(text_columns))
    if not any(r.get('event_type') for r in sample_rows):
        return text_columns

    from sklearn.metrics import mutual_info_score
    event_labels = [(r['event_type'] or 'UNKNOWN') for r in sample_rows]
    scores = []
    for c in text_columns:
        col_vals = [row.get(c) or 'NULL' for row in sample_rows]
        s = mutual_info_score(col_vals, event_labels)
        scores.append((c, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    top3 = [x[0] for x in scores[:3]]
    if 'event_type' not in top3:
        top3.append('event_type')
    return list(set(top3))

def detect_global_relevant_columns(pg_conn, tables, sample_size=10000):
    global_cols = set()
    for t in tables:
        top3 = detect_relevant_columns_table(pg_conn, t, sample_size)
        global_cols.update(top3)
    return list(global_cols)

from sklearn.metrics import mutual_info_score

def compute_global_mi_for_columns(all_data_rows, relevant_cols, min_count=1):
    valid_rows = [r for r in all_data_rows if r.get('event_type')]
    if not valid_rows:
        return {}
    column_scores = {}
    event_labels = [r['event_type'] for r in valid_rows]
    for col in relevant_cols:
        col_vals = [str(r.get(col, 'NULL')) for r in valid_rows]
        if len(set(col_vals)) <= min_count:
            column_scores[col] = 0.0
            continue
        score = mutual_info_score(col_vals, event_labels)
        column_scores[col] = score
    return column_scores

def refine_columns_by_mi(config, column_scores, min_mi=0.0001, max_cols=128, stability_threshold=10):
    """
    Uses column_score_history in config to only drop columns if their MI has been
    below min_mi for stability_threshold consecutive runs.
    """
    if 'column_score_history' not in config:
        config['column_score_history'] = {}
    history = config['column_score_history']
    stable_cols = []
    for col, score in column_scores.items():
        if score < min_mi:
            history[col] = history.get(col, 0) + 1
        else:
            history[col] = 0
        if score >= min_mi or history[col] < stability_threshold:
            stable_cols.append(col)
    if 'event_type' not in stable_cols:
        stable_cols.append('event_type')
    if 'master_columns' not in config:
        config['master_columns'] = set(stable_cols)
    else:
        config['master_columns'] = set(config['master_columns']).union(set(stable_cols))

    stable_cols = sorted(stable_cols, key=lambda c: column_scores.get(c, 0), reverse=True)
    new_cols = stable_cols[:max_cols]

    config['column_score_history'] = history
    config['master_columns']       = list(config['master_columns'])
    return new_cols


###############################################################################
# 6) Gathering Training Data => Single Model
###############################################################################
def gather_global_training_data(pg_conn, tables, relevant_cols, require_event_type=True):
    all_rows = []
    for t in tables:
        schema_part, table_part = parse_schema_and_table(t)
        actual_cols = list(get_table_columns(pg_conn, schema_part, table_part).keys())
        selected_cols = [c for c in relevant_cols if c in actual_cols]
        if not selected_cols:
            print(f"[WARN] Table {t} does not have any of the relevant columns; skipping.")
            continue
        condition = ''
        if require_event_type:
            condition = 'WHERE "event_type" IS NOT NULL AND "event_type" <> \'\''

        col_select = ', '.join(f'"{c}"' for c in selected_cols)
        q = f'''
            SELECT {col_select}
            FROM "{schema_part}"."{table_part}"
            {condition}
        '''
        with pg_conn.cursor(cursor_factory=DictCursor) as cur:
            try:
                cur.execute(q)
                rows = cur.fetchall()
                all_rows.extend(rows)
            except Exception as e:
                print(f"[WARN] gather_global_training_data: table {t} => {e}")
                pg_conn.rollback()
    return all_rows

def build_single_model(all_data_rows, relevant_cols, config):
    """
    Builds a single classifier from the combined data of all relevant tables.
    If 'customer_marked_flag' => 'YES', we'll see 'event_type=MARKED_LOGS' if set,
    which can help the model weigh these events more strongly.
    """
    if not all_data_rows:
        print("[ML] No global training data found. No model built.")
        return None, None

    from sklearn.feature_extraction.text import CountVectorizer

    X_texts = []
    Y_labels = []
    for row in all_data_rows:
        combined_text = " ".join((row.get(c) or '') for c in relevant_cols)
        X_texts.append(combined_text)
        label = row.get('event_type')
        if not label or label.strip() == "":
            label = row.get('category', 'UNKNOWN')
        Y_labels.append(label)

    vec = CountVectorizer()
    X = vec.fit_transform(X_texts)

    model_type = config.get('model_type', 'RandomForest')
    if model_type == 'NaiveBayes':
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)

    clf.fit(X, Y_labels)
    if hasattr(clf, 'feature_importances_'):
        import numpy as np
        importances = clf.feature_importances_
        feature_names = vec.get_feature_names_out()
        indices = np.argsort(importances)[::-1]
        topN = min(20, len(indices))
        print("[ML] Top feature importances:")
        for idx in indices[:topN]:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

    print(f"[ML] Single {model_type} model trained on {len(X_texts)} rows total.")
    return vec, clf


###############################################################################
# 7) Enrich a Single Table
###############################################################################
def predict_event_type(clf, vec, text):
    if not text.strip() or (clf is None or vec is None):
        return 'UNKNOWN'
    X = vec.transform([text])
    y = clf.predict(X)
    return y[0] if len(y) else 'UNKNOWN'


def update_signal_protector(pg_conn, schema_part, table_part):
    """
    Scans every text column in the given table for keywords 'ipll', 'ipvod', or 'llot' (case-insensitive).
    If any are found in a row and the event_type does not already mention 'signal_protector',
    then append ", signal_protector" to event_type.
    """
    # Get all columns and filter for text columns (e.g., 'text', 'character varying', etc.)
    cols = get_table_columns(pg_conn, schema_part, table_part)
    text_cols = [col for col, dtype in cols.items() if 'char' in dtype or 'text' in dtype]
    if not text_cols:
        print(f"[INFO] No text columns found in {schema_part}.{table_part} for signal protector update.")
        return

    # Build a WHERE clause that checks every text column for the keywords.
    conditions = []
    for col in text_cols:
        conditions.append(f'("{col}" ILIKE \'%ipll%\' OR "{col}" ILIKE \'%ipvod%\' OR "{col}" ILIKE \'%llot%\')')
    condition_clause = " OR ".join(conditions)

    # Update event_type: if already contains "signal_protector", leave it; otherwise append.
    update_sql = f'''
        UPDATE "{schema_part}"."{table_part}"
        SET "event_type" = 
            CASE 
                WHEN "event_type" ILIKE '%signal_protector%' THEN "event_type"
                ELSE 
                    COALESCE("event_type", '') ||
                    CASE WHEN "event_type" IS NULL OR "event_type" = '' THEN '' ELSE ', ' END ||
                    'signal_protector'
            END
        WHERE {condition_clause};
    '''
    with pg_conn.cursor() as cur:
        cur.execute(update_sql)
    pg_conn.commit()
    print(f"[ENRICHER] => Updated records with signal protector keywords in {schema_part}.{table_part}.")


def enrich_table(pg_conn, table_name, vectorizer, clf, relevant_cols):
    """
    - Ensures 'event_type' is set if missing (via model prediction).
    - Applies parse_data_fields() and sets 'MARKED_LOGS' where applicable.
    - NEW: Searches every text column for keywords ('ipll', 'ipvod', 'llot') and, if found,
           appends ", signal_protector" to event_type.
    """
    schema_part, table_part = parse_schema_and_table(table_name)
    refactor_table_for_enrichment(pg_conn, table_name, list(relevant_cols))
    needed_cols = set(relevant_cols) | {'id','data','event_type','timestamp','category'}

    col_select = ','.join(f'"{c}"' for c in needed_cols)
    q = f'SELECT {col_select} FROM "{schema_part}"."{table_part}"'
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(q)
        rows = cur.fetchall()

    if not rows:
        print(f"[ENRICHER] Table '{table_name}' => no rows, skipping.")
        return

    updates_pred = []
    updates_parse = []
    updates_marked_logs = []

    for r in rows:
        row_id = r['id']

        # 1) If event_type is missing, classify it
        if not r.get('event_type'):
            text_combined = " ".join((r.get(c) or '') for c in relevant_cols)
            pred = predict_event_type(clf, vectorizer, text_combined)
            if pred != 'UNKNOWN':
                updates_pred.append((pred, row_id))

        # 2) Parse data fields
        extracted = parse_data_fields(r.get('data',''))
        if extracted:
            updates_parse.append((row_id, extracted))

    # Add new columns from parsed data if necessary
    if updates_parse:
        new_cols = set()
        for _, exdict in updates_parse:
            new_cols.update(exdict.keys())
        if new_cols:
            with pg_conn.cursor() as cur_cols:
                for nc in new_cols:
                    add_col(cur_cols, schema_part, table_part, nc)
            pg_conn.commit()

    # Update with parse fields
    if updates_parse:
        with pg_conn.cursor() as curu:
            for row_id, exdict in updates_parse:
                if exdict.get('customer_marked_flag') == "YES":
                    updates_marked_logs.append((row_id, "MARKED_LOGS"))
                set_parts = []
                vals = []
                for k, v in exdict.items():
                    set_parts.append(f'"{k}"=%s')
                    vals.append(v)
                sets_clause = ",".join(set_parts)
                sql_up = f'UPDATE "{schema_part}"."{table_part}" SET {sets_clause} WHERE "id"=%s'
                vals.append(row_id)
                curu.execute(sql_up, vals)
        pg_conn.commit()
        print(f"[ENRICHER] => {len(updates_parse)} rows updated with parse fields in '{table_name}'.")

    # If flagged via parse_data_fields, override event_type to "MARKED_LOGS"
    if updates_marked_logs:
        with pg_conn.cursor() as curp:
            for (row_id, forced_label) in updates_marked_logs:
                curp.execute(f'''
                  UPDATE "{schema_part}"."{table_part}"
                  SET "event_type"=%s
                  WHERE "id"=%s
                ''', (forced_label, row_id))
        pg_conn.commit()
        print(f"[ENRICHER] => {len(updates_marked_logs)} rows labeled event_type='MARKED_LOGS' in '{table_name}'.")

    # Process predicted event types from the model
    if updates_pred:
        with pg_conn.cursor() as curp:
            up_sql = sql.SQL('UPDATE {}.{} SET "event_type"=%s WHERE "id"=%s').format(
                sql.Identifier(schema_part),
                sql.Identifier(table_part)
            )
            curp.executemany(up_sql, updates_pred)
        pg_conn.commit()
        print(f"[ENRICHER] => {len(updates_pred)} rows had event_type predicted in '{table_name}'.")

    # NEW: Now update records with signal protector keywords.
    # This function checks every text column.
    update_signal_protector(pg_conn, schema_part, table_part)


###############################################################################
# 8) Significance Heuristic
###############################################################################
def analyze_significance(pg_conn, table_name):
    """
    Basic numeric significance. Then we call highlight_customer_marked()
    to give more emphasis to any logs near customer-marked events.
    """
    schema_part, table_part = parse_schema_and_table(table_name)
    with pg_conn.cursor() as cur:
        cur.execute(f'''
            ALTER TABLE "{schema_part}"."{table_part}"
            ADD COLUMN IF NOT EXISTS freed_up_bytes_num BIGINT
        ''')
        cur.execute(f'''
            ALTER TABLE "{schema_part}"."{table_part}"
            ADD COLUMN IF NOT EXISTS malloc_occupied_num BIGINT
        ''')
        cur.execute(f'''
            ALTER TABLE "{schema_part}"."{table_part}"
            ADD COLUMN IF NOT EXISTS significance_score DOUBLE PRECISION
        ''')
    pg_conn.commit()

    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        try:
            cur.execute(f'''
              SELECT id, freed_up_bytes_num, malloc_occupied_num
              FROM "{schema_part}"."{table_part}"
            ''')
        except psycopg2.Error as e:
            print(f"[SIGNIFICANCE] Failed to fetch numeric columns from '{table_name}': {e}")
            return
        rows = cur.fetchall()
    if not rows:
        return

    updates = []
    for r in rows:
        sc = 0.0
        if r["freed_up_bytes_num"] and int(r["freed_up_bytes_num"]) > 1_000_000:
            sc += 5.0
        if r["malloc_occupied_num"] and int(r["malloc_occupied_num"]) > 12_000_000:
            sc += 3.0
        if sc > 0:
            updates.append((sc, r["id"]))
    if updates:
        with pg_conn.cursor() as cur:
            for (added_score, rowid) in updates:
                cur.execute(f'''
                  UPDATE "{schema_part}"."{table_part}"
                  SET significance_score = COALESCE(significance_score,0)::double precision + %s
                  WHERE id=%s
                ''', (added_score, rowid))
        pg_conn.commit()
        print(f"[SIGNIFICANCE] {table_name} => {len(updates)} rows updated with significance_score (numeric).")

    highlight_customer_marked(pg_conn, table_name, window_minutes=1, significance_boost=5)


def highlight_customer_marked(pg_conn, table_name, window_minutes=1, significance_boost=5):
    """
    1) Ensures we have columns => 'customer_marked_flag' TEXT, 'timestamp' TIMESTAMP.
    2) Finds all rows where customer_marked_flag='YES' and retrieves their timestamps.
    3) For each marked timestamp T, updates all logs with a timestamp within [T - window, T + window]
         to increase significance_score and set important_investigate=TRUE.
    """
    schema_part, table_part = parse_schema_and_table(table_name)

    with pg_conn.cursor() as cur:
        add_col_if_missing(cur, schema_part, table_part, "customer_marked_flag", col_type="TEXT")
        add_col_if_missing(cur, schema_part, table_part, "important_investigate", col_type="BOOLEAN")
        cur.execute(f'''
        ALTER TABLE "{schema_part}"."{table_part}"
        ADD COLUMN IF NOT EXISTS significance_score DOUBLE PRECISION
        ''')
    pg_conn.commit()

    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        try:
            cur.execute(f'''
            SELECT id, "timestamp"
            FROM "{schema_part}"."{table_part}"
            WHERE customer_marked_flag='YES'
              AND "timestamp" IS NOT NULL
            ''')
            marked_rows = cur.fetchall()
        except psycopg2.Error as e:
            print(f"[MARKED] Could not fetch customer_marked_flag from {table_name}: {e}")
            pg_conn.rollback()
            return

    if not marked_rows:
        return

    with pg_conn.cursor() as cur:
        for row in marked_rows:
            tstamp = row["timestamp"]
            if not tstamp:
                continue
            sql_update = f'''
            UPDATE "{schema_part}"."{table_part}"
            SET significance_score = COALESCE(significance_score,0)::double precision + %s,
                important_investigate = TRUE
            WHERE "timestamp" BETWEEN %s - INTERVAL '{window_minutes} minutes'
                                 AND %s + INTERVAL '{window_minutes} minutes'
            '''
            cur.execute(sql_update, (significance_boost, tstamp, tstamp))
    pg_conn.commit()

    print(f"[MARKED] => {len(marked_rows)} row(s) had 'customer_marked_flag=YES'. "
          f"Highlighted logs +/-{window_minutes} min with +{significance_boost} significance.")


###############################################################################
# 9) Master Column List Updater
###############################################################################
def update_master_columns(config, detected_cols):
    if 'master_columns' not in config:
        config['master_columns'] = set(detected_cols)
    else:
        config['master_columns'] = set(config['master_columns']).union(set(detected_cols))
    config['master_columns'].add('event_type')
    config['master_columns'] = list(config['master_columns'])
    return config['master_columns']


###############################################################################
# 10) MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Single-model enricher: one global model for all tables or for a single table."
    )
    parser.add_argument("--table_name", type=str,
        help="If set, only process one table. Otherwise, process all.")
    args = parser.parse_args()

    if not sklearn_installed:
        print("[WARN] scikit-learn not installed => no classification.")
        return

    # 1) Load credentials
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    creds_path = os.path.join(base_dir, "credentials.txt")
    if not os.path.exists(creds_path):
        print(f"[ERR] credentials.txt not found: {creds_path}")
        return
    with open(creds_path, "r") as f:
        j = json.load(f)

    db_host    = j.get("DB_HOST")
    db_name    = j.get("DB_NAME")
    db_user    = j.get("DB_USER")
    db_pass    = j.get("DB_PASSWORD")
    enrich_host= j.get("ENRICH_DB_HOST")
    enrich_db  = j.get("ENRICH_DB")
    enrich_user= j.get("ENRICH_DB_USER")
    enrich_pw  = j.get("ENRICH_DB_PASSWORD")

    # 2) Connect logs DB
    try:
        pg_conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_pass)
    except Exception as e:
        print(f"[ERR] logs DB connect fail: {e}")
        return

    # 3) Connect enrich DB
    try:
        enrich_conn = psycopg2.connect(host=enrich_host, dbname=enrich_db, user=enrich_user, password=enrich_pw)
    except Exception as e:
        print(f"[ERR] enrich DB connect fail: {e}")
        pg_conn.close()
        return

    ensure_ml_config_table(enrich_conn)
    config = load_enricher_config(enrich_conn)
    if not config:
        config = {"model_type": "RandomForest", "relevant_columns": []}

    if 'column_score_history' not in config:
        config['column_score_history'] = {}

    if args.table_name:
        single_table = args.table_name.strip()
        print(f"[MODE] Single table mode: {single_table}")

        detected_cols = detect_relevant_columns_table(pg_conn, single_table)
        master_cols = update_master_columns(config, detected_cols)
        relevant_cols = list(set(detected_cols).union(set(master_cols)))
        relevant_cols.append('event_type')
        relevant_cols = list(set(relevant_cols))
        config['relevant_columns'] = relevant_cols

        print(f"[ENRICHER] Using these relevant columns: {relevant_cols}")

        one_table_data = gather_global_training_data(pg_conn, [single_table], relevant_cols, require_event_type=True)
        if not one_table_data:
            print(f"[ENRICHER] No training data with event_type in table {single_table}. Trying no filter.")
            one_table_data = gather_global_training_data(pg_conn, [single_table], relevant_cols, require_event_type=False)

        if one_table_data:
            unique_labels = set(r.get('event_type') for r in one_table_data if r.get('event_type'))
            if len(unique_labels) < 2:
                print("[ENRICHER] Not enough event_type diversity for training => no classification.")
                vec, clf = None, None
            else:
                vec, clf = build_single_model(one_table_data, relevant_cols, config)
        else:
            print("[ENRICHER] No data found => skipping model training.")
            vec, clf = None, None

        if one_table_data:
            col_scores = compute_global_mi_for_columns(one_table_data, relevant_cols)
            new_cols   = refine_columns_by_mi(config, col_scores, min_mi=0.0001, max_cols=128)
            config['relevant_columns'] = new_cols
            store_enricher_config(enrich_conn, config)

        try:
            print(f"[ENRICHER] Enriching single table => {single_table}")
            enrich_table(pg_conn, single_table, vec, clf, config['relevant_columns'])
            analyze_significance(pg_conn, single_table)
        except psycopg2.Error as e:
            print(f"[ERROR] Something failed on table {single_table}: {e}")
            pg_conn.rollback()

        store_enricher_config(enrich_conn, config)
        pg_conn.close()
        enrich_conn.close()
        print("[ENRICHER] Done with single table mode.")
        try:
            print(f"[ENRICHER] Now gathering anomalies from {single_table}")
            subprocess.run(["python", "ingestion/linking/gather_anomalies.py",
                            "--table_name", single_table],
                            check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] gather_anomalies for {single_table} failed: {e}")

        pg_conn.close()
        enrich_conn.close()
        print("[ENRICHER] Done with single table mode.")
        return

    print("[MODE] Global (all R-tables) approach.")
    tables = get_all_user_tables(pg_conn)
    if not tables:
        print("[ENRICHER] No R####### tables found, exiting.")
        pg_conn.close()
        enrich_conn.close()
        return

    if not config.get('relevant_columns'):
        global_cols = detect_global_relevant_columns(pg_conn, tables)
        config['relevant_columns'] = global_cols
    else:
        global_cols = config['relevant_columns']

    print(f"[ENRICHER] Using these relevant columns globally: {global_cols}")

    all_train_rows = gather_global_training_data(pg_conn, tables, global_cols, require_event_type=True)
    if not all_train_rows:
        print("[ENRICHER] No event_type-labeled data found => try no filter.")
        all_train_rows = gather_global_training_data(pg_conn, tables, global_cols, require_event_type=False)

    vec, clf = build_single_model(all_train_rows, global_cols, config)
    if not vec or not clf:
        print("[ENRICHER] No single model built => no training data found. Exiting gracefully.")
        pg_conn.close()
        enrich_conn.close()
        return

    col_scores      = compute_global_mi_for_columns(all_train_rows, global_cols)
    new_global_cols = refine_columns_by_mi(config, col_scores, min_mi=0.0001, max_cols=128)
    print(f"[ENRICHER] => refined columns from {len(global_cols)} to {len(new_global_cols)}.")
    config['relevant_columns'] = new_global_cols
    store_enricher_config(enrich_conn, config)

    for t in tables:
        print(f"[ENRICHER] Enriching table => {t}")
        try:
            enrich_table(pg_conn, t, vec, clf, config['relevant_columns'])
            analyze_significance(pg_conn, t)
        except psycopg2.Error as e:
            print(f"[ERROR] Something failed on table {t}: {e}")
            pg_conn.rollback()

    store_enricher_config(enrich_conn, config)
    pg_conn.close()
    enrich_conn.close()
    print("[ENRICHER] Done - single global model used for all tables. ✅")

    try:
        print("[ENRICHER] Now gathering anomalies from all R-tables")
        subprocess.run(["python", "gather_anomalies.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] gather_anomalies (global) failed: {e}")

    pg_conn.close()
    enrich_conn.close()
    print("[ENRICHER] Done - single global model used for all tables. ✅")

if __name__ == "__main__":
    main()
