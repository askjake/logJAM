#!/usr/bin/env python3
import psycopg2
import os
import json
import sys
import argparse

def main():
    """
    Main entry point: parse args, then run gather_anomalies for either
    a specific table_name or all tables.
    """
    parser = argparse.ArgumentParser(
        description="Gather anomalies from R-tables into 'anomalies' table."
    )
    parser.add_argument("--table_name", type=str,
        help="If set, only gather anomalies from this single R########## table.")
    args = parser.parse_args()

    gather_anomalies(args.table_name)

def gather_anomalies(single_table=None):
    """
    Connect to PostgreSQL, run a DO block that:
      1) Creates any missing columns in 'anomalies' table from the R########## tables.
      2) Inserts records into 'anomalies' for rows matching:
         - lstm_is_anomaly = TRUE
         - customer_marked_flag = TRUE
         - autoenc_is_anomaly = TRUE
         - important_investigate = TRUE

      If single_table is provided, only process that table.
      Otherwise, process all R########## tables.
    """
    try:
        # Locate credentials file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        creds_path = os.path.join(base_dir, "credentials.txt")

        if not os.path.exists(creds_path):
            print(f"[ERR] credentials.txt not found: {creds_path}")
            return

        # Load credentials
        import json
        with open(creds_path, "r") as f:
            j = json.load(f)

        db_host = j.get("DB_HOST")
        db_name = j.get("DB_NAME")
        db_user = j.get("DB_USER")
        db_pass = j.get("DB_PASSWORD")

        # Connect to the logs database
        try:
            pg_conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_pass)
        except Exception as e:
            print(f"[ERR] logs DB connect fail: {e}")
            return

        pg_conn.autocommit = True
        cur = pg_conn.cursor()

        # If a single table is given, we adjust the query inside the DO block with:
        #    AND tablename = single_table
        # Otherwise, we match all R########## tables.
        single_table_clause = ""
        if single_table:
            single_table_clause = f"AND tablename = '{single_table}'"

        sql_block = rf"""
        DO $$
        DECLARE
            tbl record;
            column_list text;
            existing_columns text[];
            new_columns text[];
            new_col text;
            col_def text;
            filter_conditions text;
            lstm_exists boolean;
            cust_flag_exists boolean;
            autoenc_exists boolean;
            investigate_exists boolean;
            anomaly_count integer;
        BEGIN
            -- Fetch existing columns in the anomalies table
            SELECT array_agg(c.column_name::text)
            INTO existing_columns
            FROM information_schema.columns c
            WHERE c.table_name = 'anomalies';

            -- Iterate through matching table(s)
            FOR tbl IN
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                  AND tablename ~ '^R[0-9]{{10}}'
                  {single_table_clause}
            LOOP
                RAISE NOTICE 'Processing table: %', tbl.tablename;

                -- Identify new columns that are missing in anomalies
                SELECT array_agg(c.column_name::text)
                INTO new_columns
                FROM information_schema.columns c
                WHERE c.table_name = tbl.tablename
                  AND c.column_name NOT IN (SELECT unnest(existing_columns));

                -- Add any new columns to anomalies table
                IF new_columns IS NOT NULL THEN
                    FOR new_col IN SELECT unnest(new_columns) LOOP
                        -- Get column type from source table
                        SELECT c.data_type ||
                               CASE WHEN c.character_maximum_length IS NOT NULL
                                    THEN '(' || c.character_maximum_length || ')'
                                    ELSE ''
                               END
                        INTO col_def
                        FROM information_schema.columns c
                        WHERE c.table_name = tbl.tablename
                          AND c.column_name = new_col;

                        -- Alter anomalies table to add missing columns
                        EXECUTE format('ALTER TABLE anomalies ADD COLUMN IF NOT EXISTS %I %s', new_col, col_def);
                    END LOOP;
                END IF;

                -- Refresh existing_columns list after schema change
                SELECT array_agg(c.column_name::text)
                INTO existing_columns
                FROM information_schema.columns c
                WHERE c.table_name = 'anomalies';

                -- Generate column list dynamically
                SELECT string_agg(quote_ident(c.column_name), ', ')
                INTO column_list
                FROM information_schema.columns c
                WHERE c.table_name = tbl.tablename;

                -- Check if columns exist in this table
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = tbl.tablename AND column_name = 'lstm_is_anomaly'
                ) INTO lstm_exists;

                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = tbl.tablename AND column_name = 'customer_marked_flag'
                ) INTO cust_flag_exists;

                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = tbl.tablename AND column_name = 'autoenc_is_anomaly'
                ) INTO autoenc_exists;

                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = tbl.tablename AND column_name = 'important_investigate'
                ) INTO investigate_exists;

                RAISE NOTICE 'customer_marked_flag exists in %: %', tbl.tablename, cust_flag_exists;
                RAISE NOTICE 'lstm_is_anomaly exists in %: %', tbl.tablename, lstm_exists;
                RAISE NOTICE 'autoenc_is_anomaly exists in %: %', tbl.tablename, autoenc_exists;
                RAISE NOTICE 'important_investigate exists in %: %', tbl.tablename, investigate_exists;

                -- Build the WHERE condition
                filter_conditions := '';

                IF lstm_exists THEN
                    IF filter_conditions != '' THEN
                        filter_conditions := filter_conditions || ' OR ';
                    END IF;
                    filter_conditions := filter_conditions || 'lstm_is_anomaly = TRUE';
                END IF;

                IF cust_flag_exists THEN
                    IF filter_conditions != '' THEN
                        filter_conditions := filter_conditions || ' OR ';
                    END IF;
                    filter_conditions := filter_conditions || 'COALESCE(NULLIF(customer_marked_flag, ''''), ''false'')::BOOLEAN = TRUE';
                END IF;

                IF autoenc_exists THEN
                    IF filter_conditions != '' THEN
                        filter_conditions := filter_conditions || ' OR ';
                    END IF;
                    filter_conditions := filter_conditions || 'autoenc_is_anomaly = TRUE';
                END IF;

                IF investigate_exists THEN
                    IF filter_conditions != '' THEN
                        filter_conditions := filter_conditions || ' OR ';
                    END IF;
                    filter_conditions := filter_conditions || 'COALESCE(NULLIF(important_investigate::text, ''''), ''false'')::BOOLEAN = TRUE';
                END IF;

                IF filter_conditions IS NULL OR filter_conditions = '' THEN
                    filter_conditions := 'FALSE';
                END IF;

                -- Debug: count anomalies in the current table
                IF filter_conditions != 'FALSE' THEN
                    EXECUTE format('SELECT count(*) FROM %I WHERE %s', tbl.tablename, filter_conditions)
                    INTO anomaly_count;
                    RAISE NOTICE 'Found % anomalies in table %', anomaly_count, tbl.tablename;

                    IF anomaly_count > 0 THEN
                        EXECUTE format(
                            'INSERT INTO anomalies (table_name, %s)
                             SELECT %L, %s
                             FROM %I
                             WHERE %s
                             ORDER BY timestamp DESC',
                            column_list, tbl.tablename, column_list, tbl.tablename, filter_conditions
                        );
                        RAISE NOTICE 'Inserted anomalies from table %', tbl.tablename;
                    ELSE
                        RAISE NOTICE 'No anomalies to insert from table %', tbl.tablename;
                    END IF;
                END IF;
            END LOOP;
        END $$;
        """

        # Execute the block
        cur.execute(sql_block)
        cur.close()
        pg_conn.close()
        print("Anomalies gathered and inserted into 'anomalies' table successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
