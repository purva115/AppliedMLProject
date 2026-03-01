"""
Dataset Preparation Script
==========================
Auto-downloads the Kaggle EDA | 11,000 Medicines dataset via kagglehub
and converts it into a JSONL file ready for TinyLlama fine-tuning.

Usage (auto-download):
    python prepare_dataset.py

Usage (manual CSV already downloaded):
    python prepare_dataset.py --input data/raw/medicines.csv

Kaggle dataset: https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details
"""
import argparse
import glob
import json
import logging
import os
import random
import shutil

import kagglehub
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Columns of interest ────────────────────────────────────────────────────────
REQUIRED_COLS = {
    "Medicine Name", "Composition", "Uses", "Side_effects", "Excellent Review %",
    "Average Review %", "Poor Review %",
}


def load_csv(path: str) -> pd.DataFrame:
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def compute_satisfaction(row) -> float:
    """Return user satisfaction % = Excellent + 0.5 × Average."""
    return float(row.get("Excellent Review %", 0)) + 0.5 * float(row.get("Average Review %", 0))


def row_to_text(row) -> str:
    """Convert a medicine row to a natural-language training sentence."""
    name = row["Medicine Name"].strip()
    composition = row["Composition"].strip()
    uses = row["Uses"].strip()
    side_effects = row["Side_effects"].strip()
    satisfaction = compute_satisfaction(row)

    rating = (
        "Excellent" if satisfaction >= 80 else
        "Good"      if satisfaction >= 60 else
        "Average"
    )

    return (
        f"{name} is composed of {composition}. "
        f"It is used for: {uses}. "
        f"Common side effects include: {side_effects}. "
        f"User satisfaction is rated as {rating} ({satisfaction:.1f}%)."
    )


def prepare(input_path: str, output_path: str, min_satisfaction: float = 70.0,
            val_split: float = 0.1, seed: int = 42):
    df = load_csv(input_path)
    logger.info("Rows before filtering: %d", len(df))

    # Compute satisfaction and filter
    df["satisfaction"] = df.apply(compute_satisfaction, axis=1)
    df = df[df["satisfaction"] >= min_satisfaction].copy()
    logger.info("Rows after satisfaction filter (>= %.1f%%): %d", min_satisfaction, len(df))

    # Drop irrelevant columns
    df = df.drop(columns=["Image URL", "Manufacturer"] if "Image URL" in df.columns else [], errors="ignore")

    # Build text records
    records = [{"text": row_to_text(row)} for _, row in df.iterrows()]
    random.seed(seed)
    random.shuffle(records)

    # Split train / val
    split_idx = int(len(records) * (1 - val_split))
    train_records = records[:split_idx]
    val_records   = records[split_idx:]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    train_path = output_path
    val_path   = output_path.replace(".jsonl", "_val.jsonl")

    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")
        logger.info("Saved %d records → %s", len(data), path)

    write_jsonl(train_path, train_records)
    write_jsonl(val_path,   val_records)
    logger.info("Done! Train: %d | Val: %d", len(train_records), len(val_records))


def download_dataset(raw_dir: str = "data/raw") -> str:
    """
    Downloads the Kaggle dataset via kagglehub and copies the CSV into
    raw_dir.  Returns the path to the CSV file.
    """
    logger.info("Downloading dataset via kagglehub …")
    kaggle_path = kagglehub.dataset_download("singhnavjot2062001/11000-medicine-details")
    logger.info("Dataset downloaded to: %s", kaggle_path)

    # Find the CSV inside the downloaded folder
    csv_files = glob.glob(os.path.join(kaggle_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {kaggle_path}")

    src_csv = csv_files[0]
    os.makedirs(raw_dir, exist_ok=True)
    dest_csv = os.path.join(raw_dir, "medicines.csv")
    shutil.copy2(src_csv, dest_csv)
    logger.info("CSV copied to %s", dest_csv)
    return dest_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PharmaLLM training dataset")
    parser.add_argument(
        "--input", default=None,
        help="Path to medicines.csv. If omitted, dataset is auto-downloaded via kagglehub."
    )
    parser.add_argument("--output", default="data/processed/medicines_train.jsonl")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Directory to store the downloaded CSV.")
    parser.add_argument("--min-satisfaction", type=float, default=70.0)
    parser.add_argument("--val-split",        type=float, default=0.1)
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    # Auto-download if no --input given
    input_path = args.input
    if input_path is None:
        input_path = download_dataset(raw_dir=args.raw_dir)

    prepare(
        input_path=input_path,
        output_path=args.output,
        min_satisfaction=args.min_satisfaction,
        val_split=args.val_split,
        seed=args.seed,
    )
