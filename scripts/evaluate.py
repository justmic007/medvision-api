"""Evaluate the TorchXRayVision model on a labeled NIH sample (Phase 8).

Computes per-pathology ROC/AUC: how well the model distinguishes each finding
present vs absent, independent of any threshold (D-07). Runs on a random subset
for tractable CPU runtime; AUC is only reported for findings with enough
positive examples (rare findings give unstable AUCs on small samples).

Usage:  python -m scripts.evaluate [--n 5606] [--min-pos 30]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from app.services.classifier import ChestXrayClassifier
from app.services.preprocessing import ChestXrayPreprocessor

IMAGES_DIR = Path("data/nih-eval/sample/images")
LABELS_CSV = Path("data/nih-eval/sample_labels.csv")
OUT_JSON = Path("data/nih-eval/evaluation_results.json")


def main(n: int, min_pos: int, seed: int) -> None:
    random.seed(seed)

    clf = ChestXrayClassifier()
    pre = ChestXrayPreprocessor()
    pathologies = clf.pathologies

    df = pd.read_csv(LABELS_CSV)
    df = df[df["Image Index"].apply(lambda f: (IMAGES_DIR / f).exists())]
    if n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"Evaluating on {len(df)} images...")

    y_true = np.zeros((len(df), len(pathologies)), dtype=int)
    y_score = np.zeros((len(df), len(pathologies)), dtype=float)

    for i, row in df.iterrows():
        findings = set(str(row["Finding Labels"]).split("|"))
        for j, path in enumerate(pathologies):
            y_true[i, j] = 1 if path in findings else 0

        tensor = pre.process(IMAGES_DIR / row["Image Index"])
        result = clf.predict(tensor)
        probs = {f.name: f.probability for f in result.findings}
        for j, path in enumerate(pathologies):
            y_score[i, j] = probs.get(path, 0.0)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)} processed")

    results = {}
    print("\n=== Per-pathology AUC ===")
    for j, path in enumerate(pathologies):
        pos = int(y_true[:, j].sum())
        neg = len(df) - pos
        if pos < min_pos or neg < min_pos:
            results[path] = {"auc": None, "n_pos": pos, "reason": "insufficient examples"}
            print(f"  {path:28s}  skipped (pos={pos})")
            continue
        auc = roc_auc_score(y_true[:, j], y_score[:, j])
        results[path] = {"auc": round(float(auc), 4), "n_pos": pos}
        print(f"  {path:28s}  AUC {auc:.3f}  (pos={pos})")

    aucs = [r["auc"] for r in results.values() if r["auc"] is not None]
    summary = {
        "n_images": len(df),
        "min_pos_threshold": min_pos,
        "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        "model": clf.model_name,
        "per_pathology": results,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nMean AUC (reported findings): {summary['mean_auc']}")
    print(f"Saved -> {OUT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--min-pos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n, args.min_pos, args.seed)
