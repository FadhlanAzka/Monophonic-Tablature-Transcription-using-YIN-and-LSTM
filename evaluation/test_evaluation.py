"""
evaluate_lstm_testset.py

Evaluasi model LSTM pada TEST SET (folder terpisah):
- Dataset test dipilih via Tkinter
- Tidak ada split (100% test)
- Hitung metrik token-based
- Simpan:
    - summary.json
    - tp_fp_fn.json (untuk hitungan manual)
    - summary.png
"""

import os
import json
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


# =============================================================================
# TKINTER PATH SELECTION
# =============================================================================
def select_paths_test_eval():
    root = tk.Tk()
    root.withdraw()

    print("Pilih RUN_DIR (artifacts run)...")
    run_dir = filedialog.askdirectory(title="Pilih RUN_DIR")
    if not run_dir:
        raise SystemExit("RUN_DIR tidak dipilih")

    config_path = os.path.join(run_dir, "meta", "training_config.json")
    if not os.path.exists(config_path):
        config_path = filedialog.askopenfilename(
            title="Pilih training_config.json",
            filetypes=[("JSON files", "*.json")]
        )
    with open(config_path, "r") as f:
        training_config = json.load(f)

    print("Pilih FOLDER DATASET TEST...")
    test_dataset_dir = filedialog.askdirectory(title="Pilih folder DATASET TEST")
    if not test_dataset_dir:
        raise SystemExit("Dataset test tidak dipilih")

    print("Pilih MODEL FILE...")
    model_path = filedialog.askopenfilename(
        title="Pilih file model",
        filetypes=[("Model files", "*.ckpt *.pt *.pth *.jit *.ts.pt")]
    )
    if not model_path:
        raise SystemExit("Model tidak dipilih")

    print("Pilih folder untuk SAVE HASIL EVALUASI...")
    save_dir = filedialog.askdirectory(title="Pilih folder save eval result")
    if not save_dir:
        raise SystemExit("Folder save tidak dipilih")

    return run_dir, training_config, test_dataset_dir, model_path, save_dir


# =============================================================================
# DATASET
# =============================================================================
class MidiTokenSequenceDataset(Dataset):
    def __init__(self, df, seq_len, seq_hop, midi_col, token_col, file_col):
        self.seq_len = seq_len
        self.seq_hop = seq_hop

        X, Y = [], []
        for _, g in df.groupby(file_col):
            midi = g[midi_col].to_numpy()
            tok = g[token_col].to_numpy()

            if len(midi) < seq_len:
                continue

            for i in range(0, len(midi) - seq_len + 1, seq_hop):
                X.append(midi[i:i+seq_len])
                Y.append(tok[i:i+seq_len])

        self.X = np.stack(X)
        self.Y = np.stack(Y)

        print(f"[TEST DATASET] {len(self.X)} sequences")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.Y[idx], dtype=torch.long),
        )


# =============================================================================
# MODEL
# =============================================================================
class NaiveTabMapperLSTM(nn.Module):
    def __init__(self, midi_vocab_size, num_classes,
                 midi_embedding_dim, hidden_size,
                 num_layers, bidirectional, dropout):
        super().__init__()
        self.embed = nn.Embedding(midi_vocab_size, midi_embedding_dim)
        self.lstm = nn.LSTM(
            midi_embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.fc(x)


# =============================================================================
# MAIN
# =============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir, cfg, test_dir, model_path, save_dir = select_paths_test_eval()

    hp = cfg["hyperparameters"]
    SEQ_LEN = hp["SEQ_LEN"]
    SEQ_HOP = hp["SEQ_HOP"]
    BATCH_SIZE = hp["BATCH_SIZE"]

    num_classes = cfg["num_classes"]
    midi_vocab_size = cfg["midi_vocab_size"]

    # --- Load TEST CSV ---
    csvs = glob.glob(os.path.join(test_dir, "*.csv"))
    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        df["file"] = os.path.basename(p)
        dfs.append(df)
    df_test = pd.concat(dfs, ignore_index=True)

    test_dataset = MidiTokenSequenceDataset(
        df_test, SEQ_LEN, SEQ_HOP,
        midi_col="midi",
        token_col="token_idx",
        file_col="file"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --- Load model ---
    def load_model_flexible(model_path, device, cfg):
        print("Loading model:", model_path)

        # 1) TorchScript (jit)
        if model_path.endswith((".jit", ".ts.pt")):
            print("Detected TorchScript model")
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
            return model

        # 2) torch.load biasa (checkpoint / state_dict)
        ckpt = torch.load(
            model_path,
            map_location=device,
            weights_only=False  # <<< PENTING UNTUK PYTORCH 2.6
        )

        model = NaiveTabMapperLSTM(
            cfg["midi_vocab_size"],
            cfg["num_classes"],
            cfg["hyperparameters"]["MIDI_EMBED_DIM"],
            cfg["hyperparameters"]["HIDDEN_SIZE"],
            cfg["hyperparameters"]["NUM_LAYERS"],
            cfg["hyperparameters"]["BIDIRECTIONAL"],
            cfg["hyperparameters"]["DROPOUT"],
        ).to(device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.eval()
        return model


    model = load_model_flexible(
        model_path=model_path,
        device=device,
        cfg=cfg
    )

    # --- Inference ---
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(-1).cpu().numpy()

            y_true.append(y.numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true).reshape(-1)
    y_pred = np.concatenate(y_pred).reshape(-1)

    # =============================================================================
    # METRICS
    # =============================================================================
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "num_tokens_test": int(len(y_true))
    }

    # =============================================================================
    # TP FP FN (TOKEN-BASED)
    # =============================================================================
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    tp_fp_fn = {}
    for i in range(num_classes):
        TP = int(cm[i, i])
        FP = int(cm[:, i].sum() - TP)
        FN = int(cm[i, :].sum() - TP)

        tp_fp_fn[str(i)] = {
            "TP": TP,
            "FP": FP,
            "FN": FN
        }

    tp_fp_fn["TOTAL"] = {
        "TP": int(np.trace(cm)),
        "FP": int(cm.sum() - np.trace(cm)),
        "FN": int(cm.sum() - np.trace(cm))
    }

    # =============================================================================
    # SAVE
    # =============================================================================
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(save_dir, "tp_fp_fn.json"), "w") as f:
        json.dump(tp_fp_fn, f, indent=2)

    # Plot
    metric_names = ["accuracy", "precision", "recall", "f1"]

    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    ]

    x = np.arange(len(metric_names))

    plt.figure(figsize=(10, 5))
    plt.bar(x, values, width=0.6, color="tab:blue")
    plt.xticks(x, metric_names, rotation=0, ha="right")

    # --- Zoom otomatis Y-axis ---
    min_v = min(values)
    max_v = max(values)
    padding = (max_v - min_v) * 0.2 if max_v != min_v else 0.05
    plt.ylim(min_v - padding, max_v + padding)

    # --- Tampilkan angka value di atas bar ---
    for i, v in enumerate(values):
        plt.text(i, v + padding * 0.1, f"{v:.4f}", ha="center", fontsize=10)

    # --- Grid horizontal ---
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.ylabel("Score")
    plt.title("")
    plt.tight_layout()

    summary_png_path = os.path.join(save_dir, "summary.png")
    plt.savefig(summary_png_path, dpi=150)
    plt.close()

    print("=== TEST EVALUATION DONE ===")
    print("Saved to:", save_dir)


if __name__ == "__main__":
    main()
