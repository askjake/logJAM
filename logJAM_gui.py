#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import subprocess
import sys
import os

class LogJamGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("LogJAM GUI Dashboard")
        self.geometry("800x600")

        # Create a Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        # 1) Ingest Tab
        self.tab_ingest = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ingest, text="1) Ingest")

        # 2) Gather Anomalies Tab
        self.tab_gather = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_gather, text="2) Gather Anomalies")

        # 3) Enricher Tab
        self.tab_enricher = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_enricher, text="3) Enricher")

        # 4) ML Trainer Tab
        self.tab_ml_trainer = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ml_trainer, text="4) ML Trainer")

        # 5) Autoencoder Tab
        self.tab_autoencoder = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_autoencoder, text="5) Autoencoder")

        # 6) LSTM Anomaly Tab
        self.tab_lstm = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_lstm, text="6) LSTM Anomaly")

        # 7) 3D Plot Tab
        self.tab_3dplot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_3dplot, text="7) Plot 3D")

        # Build each tab’s widgets
        self.build_tab_ingest()
        self.build_tab_gather()
        self.build_tab_enricher()
        self.build_tab_ml_trainer()
        self.build_tab_autoencoder()
        self.build_tab_lstm()
        self.build_tab_3dplot()

    # ---------------------------
    # Tab 1: Ingest
    # ---------------------------
    def build_tab_ingest(self):
        frm = self.tab_ingest

        # Directory label + entry
        lbl_dir = ttk.Label(frm, text="Ingest Directory:")
        lbl_dir.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.var_ingest_dir = tk.StringVar()
        ent_dir = ttk.Entry(frm, textvariable=self.var_ingest_dir, width=40)
        ent_dir.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # “Select Dir” button
        def select_dir():
            d = filedialog.askdirectory()
            if d:
                self.var_ingest_dir.set(d)
        btn_browse = ttk.Button(frm, text="Browse...", command=select_dir)
        btn_browse.grid(row=0, column=2, padx=5, pady=5)

        # Workers label + entry
        lbl_workers = ttk.Label(frm, text="Number of workers:")
        lbl_workers.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.var_workers = tk.StringVar(value="4")
        ent_workers = ttk.Entry(frm, textvariable=self.var_workers, width=5)
        ent_workers.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # “Run Ingest” button
        def run_ingest():
            directory = self.var_ingest_dir.get().strip()
            workers = self.var_workers.get().strip()

            if not directory:
                tk.messagebox.showerror("Error", "Please select an ingest directory.")
                return

            cmd = [
                sys.executable,
                os.path.join("ingestion", "log_ingest.py"),
                "--directory", directory,
                "--workers", workers
            ]
            subprocess.run(cmd)
            tk.messagebox.showinfo("Ingest", "Log ingestion completed!")

        btn_run = ttk.Button(frm, text="Run Ingest", command=run_ingest)
        btn_run.grid(row=2, column=1, padx=5, pady=10)

    # ---------------------------
    # Tab 2: Gather Anomalies
    # ---------------------------
    def build_tab_gather(self):
        frm = self.tab_gather

        lbl_info = ttk.Label(frm, text="Gather anomalies from R-tables into 'anomalies' table.")
        lbl_info.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        def run_gather():
            cmd = [
                sys.executable,
                os.path.join("ingestion", "linking", "gather_anomalies.py")
            ]
            subprocess.run(cmd)
            tk.messagebox.showinfo("Gather Anomalies", "Anomalies gathered successfully!")

        btn_run = ttk.Button(frm, text="Run Gather Anomalies", command=run_gather)
        btn_run.grid(row=1, column=0, pady=10)

    # ---------------------------
    # Tab 3: Enricher
    # ---------------------------
    def build_tab_enricher(self):
        frm = self.tab_enricher
        lbl_table = ttk.Label(frm, text="Table Name (optional):")
        lbl_table.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.var_table = tk.StringVar()
        ent_table = ttk.Entry(frm, textvariable=self.var_table, width=20)
        ent_table.grid(row=0, column=1, padx=5, pady=5)

        def run_enricher():
            table_name = self.var_table.get().strip()
            cmd = [
                sys.executable,
                os.path.join("ingestion", "linking", "enricher.py")
            ]
            if table_name:
                cmd.extend(["--table_name", table_name])

            subprocess.run(cmd)
            tk.messagebox.showinfo("Enricher", "Enrichment completed!")

        btn_run = ttk.Button(frm, text="Run Enricher", command=run_enricher)
        btn_run.grid(row=1, column=1, pady=10)

    # ---------------------------
    # Tab 4: ML Trainer
    # ---------------------------
    def build_tab_ml_trainer(self):
        frm = self.tab_ml_trainer
        lbl_info = ttk.Label(frm, text="Train single global model on anomaly data.")
        lbl_info.grid(row=0, column=0, padx=5, pady=5)

        def run_ml_trainer():
            cmd = [
                sys.executable,
                os.path.join("ingestion", "linking", "ml_trainer.py")
            ]
            subprocess.run(cmd)
            tk.messagebox.showinfo("ML Trainer", "ML training completed!")

        btn_run = ttk.Button(frm, text="Run ML Trainer", command=run_ml_trainer)
        btn_run.grid(row=1, column=0, pady=10)

    # ---------------------------
    # Tab 5: Autoencoder
    # ---------------------------
    def build_tab_autoencoder(self):
        frm = self.tab_autoencoder

        # Table Name
        lbl_table = ttk.Label(frm, text="Table Name:")
        lbl_table.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.var_ae_table = tk.StringVar()
        ent_table = ttk.Entry(frm, textvariable=self.var_ae_table, width=20)
        ent_table.grid(row=0, column=1, pady=5, sticky="w")

        # Model Path
        lbl_model_path = ttk.Label(frm, text="Model Path:")
        lbl_model_path.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.var_ae_model_path = tk.StringVar()
        ent_model_path = ttk.Entry(frm, textvariable=self.var_ae_model_path, width=20)
        ent_model_path.grid(row=1, column=1, pady=5, sticky="w")

        # Latent Dim
        lbl_latent = ttk.Label(frm, text="Latent Dim:")
        lbl_latent.grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.var_ae_latent_dim = tk.StringVar(value="16")
        ent_latent = ttk.Entry(frm, textvariable=self.var_ae_latent_dim, width=6)
        ent_latent.grid(row=2, column=1, pady=5, sticky="w")

        # Epochs
        lbl_epochs = ttk.Label(frm, text="Epochs:")
        lbl_epochs.grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.var_ae_epochs = tk.StringVar(value="10")
        ent_epochs = ttk.Entry(frm, textvariable=self.var_ae_epochs, width=6)
        ent_epochs.grid(row=3, column=1, pady=5, sticky="w")

        # Batch Size
        lbl_bs = ttk.Label(frm, text="Batch Size:")
        lbl_bs.grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.var_ae_batch_size = tk.StringVar(value="64")
        ent_bs = ttk.Entry(frm, textvariable=self.var_ae_batch_size, width=6)
        ent_bs.grid(row=4, column=1, pady=5, sticky="w")

        # Start Time / End Time
        lbl_start = ttk.Label(frm, text="Start Time (YYYY-MM-DD):")
        lbl_start.grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.var_ae_start = tk.StringVar()
        ent_start = ttk.Entry(frm, textvariable=self.var_ae_start, width=10)
        ent_start.grid(row=5, column=1, pady=5, sticky="w")

        lbl_end = ttk.Label(frm, text="End Time (YYYY-MM-DD):")
        lbl_end.grid(row=6, column=0, sticky="e", padx=5, pady=5)
        self.var_ae_end = tk.StringVar()
        ent_end = ttk.Entry(frm, textvariable=self.var_ae_end, width=10)
        ent_end.grid(row=6, column=1, pady=5, sticky="w")

        # Plot checkbox
        self.var_ae_plot = tk.BooleanVar()
        chk_plot = ttk.Checkbutton(frm, text="Generate Plot?", variable=self.var_ae_plot)
        chk_plot.grid(row=7, column=1, sticky="w")

        def run_autoencoder():
            args = [
                sys.executable,
                os.path.join("analysis", "anomaly_detection", "autoencoder.py"),
                "--table_name", self.var_ae_table.get(),
                "--model_path", self.var_ae_model_path.get(),
                "--latent_dim", self.var_ae_latent_dim.get(),
                "--epochs", self.var_ae_epochs.get(),
                "--batch_size", self.var_ae_batch_size.get()
            ]
            if self.var_ae_start.get():
                args.extend(["--start_time", self.var_ae_start.get()])
            if self.var_ae_end.get():
                args.extend(["--end_time", self.var_ae_end.get()])
            if self.var_ae_plot.get():
                args.append("--plot")

            subprocess.run(args)
            tk.messagebox.showinfo("Autoencoder", "Autoencoder script finished!")

        btn_run = ttk.Button(frm, text="Run Autoencoder", command=run_autoencoder)
        btn_run.grid(row=8, column=1, pady=10, sticky="e")

    # ---------------------------
    # Tab 6: LSTM Anomaly
    # ---------------------------
    def build_tab_lstm(self):
        frm = self.tab_lstm

        # Table Name
        lbl_table = ttk.Label(frm, text="Table Name:")
        lbl_table.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.var_lstm_table = tk.StringVar()
        ent_table = ttk.Entry(frm, textvariable=self.var_lstm_table, width=20)
        ent_table.grid(row=0, column=1, pady=5, sticky="w")

        # Model Path
        lbl_model_path = ttk.Label(frm, text="Model Path:")
        lbl_model_path.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.var_lstm_model_path = tk.StringVar()
        ent_model_path = ttk.Entry(frm, textvariable=self.var_lstm_model_path, width=20)
        ent_model_path.grid(row=1, column=1, pady=5, sticky="w")

        # Sequence Length
        lbl_seq_len = ttk.Label(frm, text="Sequence Length:")
        lbl_seq_len.grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.var_lstm_seq_len = tk.StringVar(value="50")
        ent_seq_len = ttk.Entry(frm, textvariable=self.var_lstm_seq_len, width=6)
        ent_seq_len.grid(row=2, column=1, pady=5, sticky="w")

        # Epochs
        lbl_epochs = ttk.Label(frm, text="Epochs:")
        lbl_epochs.grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.var_lstm_epochs = tk.StringVar(value="10")
        ent_epochs = ttk.Entry(frm, textvariable=self.var_lstm_epochs, width=6)
        ent_epochs.grid(row=3, column=1, pady=5, sticky="w")

        # Hidden Size
        lbl_hidden = ttk.Label(frm, text="Hidden Size:")
        lbl_hidden.grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.var_lstm_hidden_size = tk.StringVar(value="128")
        ent_hidden = ttk.Entry(frm, textvariable=self.var_lstm_hidden_size, width=6)
        ent_hidden.grid(row=4, column=1, pady=5, sticky="w")

        # Plot checkbox
        self.var_lstm_plot = tk.BooleanVar()
        chk_plot = ttk.Checkbutton(frm, text="Generate Plot?", variable=self.var_lstm_plot)
        chk_plot.grid(row=5, column=1, sticky="w")

        def run_lstm():
            args = [
                sys.executable,
                os.path.join("analysis", "anomaly_detection", "lstm_anomaly.py"),
                "--table_name", self.var_lstm_table.get(),
                "--model_path", self.var_lstm_model_path.get(),
                "--seq_len", self.var_lstm_seq_len.get(),
                "--epochs", self.var_lstm_epochs.get(),
                "--hidden_size", self.var_lstm_hidden_size.get()
            ]
            if self.var_lstm_plot.get():
                args.append("--plot")

            subprocess.run(args)
            tk.messagebox.showinfo("LSTM", "LSTM Anomaly script finished!")

        btn_run = ttk.Button(frm, text="Run LSTM Anomaly", command=run_lstm)
        btn_run.grid(row=6, column=1, pady=10, sticky="e")

    # ---------------------------
    # Tab 7: 3D Plot
    # ---------------------------
    def build_tab_3dplot(self):
        frm = self.tab_3dplot

        lbl_info = ttk.Label(frm, text="Run 3D Analysis (3d_analysis.py) or Plot Embeddings (embed_and_plot.py)")
        lbl_info.grid(row=0, column=0, padx=5, pady=5, columnspan=2)

        # Input fields for filtering
        self.var_plot_start = tk.StringVar()
        self.var_plot_end = tk.StringVar()
        self.var_model = tk.StringVar(value="all-MiniLM-L6-v2")  # Default model selection

        lbl_start = ttk.Label(frm, text="Start Date (YYYY-MM-DD):")
        lbl_start.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ent_start = ttk.Entry(frm, textvariable=self.var_plot_start, width=12)
        ent_start.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        lbl_end = ttk.Label(frm, text="End Date (YYYY-MM-DD):")
        lbl_end.grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ent_end = ttk.Entry(frm, textvariable=self.var_plot_end, width=12)
        ent_end.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        lbl_model = ttk.Label(frm, text="Sentence Transformer Model:")
        lbl_model.grid(row=3, column=0, sticky="e", padx=5, pady=5)
        ent_model = ttk.Entry(frm, textvariable=self.var_model, width=20)
        ent_model.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # ---------------------------
        # Run 3D Analysis Button
        # ---------------------------
        def run_3d_analysis():
            args = [
                sys.executable,
                os.path.join("analysis", "anomaly_detection", "3d_analysis.py")
            ]
            if self.var_plot_start.get():
                args.extend(["--start_date", self.var_plot_start.get()])
            if self.var_plot_end.get():
                args.extend(["--end_date", self.var_plot_end.get()])
            subprocess.run(args)
            messagebox.showinfo("3D Analysis", "3D analysis completed!")

        btn_run_3d_analysis = ttk.Button(frm, text="Run 3D Analysis", command=run_3d_analysis)
        btn_run_3d_analysis.grid(row=4, column=1, pady=10, sticky="e")

        # ---------------------------
        # Run Embedding & Plotting Button
        # ---------------------------
        def run_embed_and_plot():
            args = [
                sys.executable,
                os.path.join("analysis", "visualization", "embed_and_plot.py")
            ]
            if self.var_plot_start.get():
                args.extend(["--start_date", self.var_plot_start.get()])
            if self.var_plot_end.get():
                args.extend(["--end_date", self.var_plot_end.get()])
            if self.var_model.get():
                args.extend(["--model", self.var_model.get()])
            subprocess.run(args)
            messagebox.showinfo("Embed & Plot", "Embedding & Plotting completed!")

        btn_run_embed_plot = ttk.Button(frm, text="Run Embed & Plot", command=run_embed_and_plot)
        btn_run_embed_plot.grid(row=5, column=1, pady=10, sticky="e")

def main():
    app = LogJamGUI()
    app.mainloop()

if __name__ == "__main__":
    main()