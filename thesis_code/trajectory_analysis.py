"""
Analiza przebiegów uczenia — metryki z checkpointów do tekstu pracy (rozdz. 6).

Użycie:
  python -m thesis_code.trajectory_analysis
  python -m thesis_code.trajectory_analysis --seed 42 --dataset CIFAR100
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

from .config import CFG
from .storage import get_results_path, load_checkpoint, setup_paths

METHOD_PREFIXES: List[Tuple[str, str]] = [
    ("Early Stopping", "es"),
    ("SGDR", "sgdr"),
    ("SWA", "swa"),
    ("Noise Injection", "ni"),
    ("MU Escape", "mu"),
]


def _peak_epoch(values: List[float]) -> Tuple[int, float]:
    if not values:
        return 0, 0.0
    idx = max(range(len(values)), key=lambda i: values[i])
    return idx + 1, float(values[idx])


def summarize_seed_trajectories(seed: int) -> Dict[str, Any]:
    """Zbiera statystyki val_acc / train_loss / ostrość dla jednego ziarna."""
    rows: Dict[str, Any] = {"seed": seed, "dataset": CFG["dataset"], "methods": {}}

    for name, prefix in METHOD_PREFIXES:
        ckpt = load_checkpoint(f"{prefix}_seed{seed}")
        if ckpt is None:
            continue
        h = ckpt.get("history", {})
        val = [float(x) for x in h.get("val_acc", [])]
        loss = [float(x) for x in h.get("train_loss", [])]
        peak_ep, peak_val = _peak_epoch(val)
        entry: Dict[str, Any] = {
            "epochs": len(val),
            "peak_val_acc": peak_val,
            "peak_val_epoch": peak_ep,
            "final_val_acc": val[-1] if val else None,
            "final_train_loss": loss[-1] if loss else None,
            "test_acc": float(h.get("test_acc", 0) or 0),
            "sharpness": float(h.get("sharpness", 0) or 0),
        }
        if name == "MU Escape":
            entry["sharp_before"] = h.get("sharp_before")
            entry["sharp_after_ul"] = h.get("sharp_after_ul")
        rows["methods"][name] = entry

    return rows


def export_trajectory_summary(seed: int) -> str:
    """Zapisuje JSON w results/{dataset}/trajektorie_seed{seed}.json."""
    data = summarize_seed_trajectories(seed)
    path = get_results_path(f"trajektorie_seed{seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksport metryk przebiegów uczenia")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], default=None)
    args = parser.parse_args()

    setup_paths()
    if args.dataset:
        CFG["dataset"] = args.dataset

    path = export_trajectory_summary(args.seed)
    data = summarize_seed_trajectories(args.seed)
    print(f"Dataset: {CFG['dataset']}, seed: {args.seed}")
    print(f"Zapisano: {path}\n")
    for name, m in data["methods"].items():
        print(
            f"{name:18s} | ep={m['epochs']:2d} | "
            f"peak_val={m['peak_val_acc']:.4f}@{m['peak_val_epoch']:2d} | "
            f"test={m['test_acc']:.4f} | sharp={m['sharpness']:.4f}"
        )


if __name__ == "__main__":
    main()
