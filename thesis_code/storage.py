"""
Zapis i odczyt checkpointów — Colab (Drive) i lokalnie.

Checkpointy i wyniki w podfolderach per dataset:
  checkpoints/CIFAR10/es_seed42.pt
  results/CIFAR10/tabela_finalna.csv
  checkpoints/CIFAR100/...
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

import torch

from . import config as cfg

try:
    import google.colab  # type: ignore

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DRIVE_DIR: Optional[str] = None


def _dataset_name() -> str:
    return cfg.CFG.get("dataset", "CIFAR10")


def _ensure_dataset_dir(base: str) -> str:
    path = os.path.join(base, _dataset_name())
    os.makedirs(path, exist_ok=True)
    return path


def ensure_dataset_dirs() -> None:
    """Tworzy podfoldery checkpoints/ i results/ dla bieżącego CFG['dataset']."""
    _ensure_dataset_dir(cfg.CHECKPOINT_DIR)
    _ensure_dataset_dir(cfg.RESULTS_DIR)


def setup_paths(
    project_root: str | None = None,
    drive_subdir: str = "thesis_sharp_minima",
) -> str:
    """Inicjalizuje katalogi projektu (+ Drive w Colab)."""
    global DRIVE_DIR
    from . import config as cfg_mod

    root = project_root or cfg.PROJECT_ROOT
    ckpt = os.path.join(root, "checkpoints")
    res = os.path.join(root, "results")
    cfg_mod.CHECKPOINT_DIR = ckpt
    cfg_mod.RESULTS_DIR = res
    os.makedirs(ckpt, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    ensure_dataset_dirs()

    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive")
        DRIVE_DIR = f"/content/drive/MyDrive/{drive_subdir}"
        for sub in ("checkpoints", "results"):
            os.makedirs(os.path.join(DRIVE_DIR, sub, _dataset_name()), exist_ok=True)
        print(f"Colab Drive: {DRIVE_DIR}")
        print(f"Dataset folder: {_dataset_name()}")

    return root


def save_to_drive(local_path: str, rel_path: str | None = None) -> None:
    if not IN_COLAB or DRIVE_DIR is None:
        return
    rel_path = rel_path or os.path.relpath(local_path, cfg.PROJECT_ROOT)
    dst = os.path.join(DRIVE_DIR, rel_path.replace("\\", "/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(local_path, dst)
    print(f"  Drive: {dst}")


def save_checkpoint(
    state_dict: Dict[str, Any],
    name: str,
    history: Optional[Dict] = None,
) -> str:
    payload: Dict[str, Any] = {"state_dict": state_dict, "dataset": _dataset_name()}
    if history is not None:
        payload["history"] = history

    local = os.path.join(_ensure_dataset_dir(cfg.CHECKPOINT_DIR), f"{name}.pt")
    torch.save(payload, local)
    save_to_drive(local, f"checkpoints/{_dataset_name()}/{name}.pt")
    return local


def load_checkpoint(name: str) -> Optional[Dict[str, Any]]:
    local = os.path.join(_ensure_dataset_dir(cfg.CHECKPOINT_DIR), f"{name}.pt")

    if IN_COLAB and DRIVE_DIR is not None:
        drive_path = os.path.join(
            DRIVE_DIR, "checkpoints", _dataset_name(), f"{name}.pt"
        )
        # Stara ścieżka bez podfolderu dataset (kompatybilność wsteczna)
        drive_legacy = os.path.join(DRIVE_DIR, "checkpoints", f"{name}.pt")
        if os.path.exists(drive_path):
            shutil.copy2(drive_path, local)
            print(f"  Wczytano z Drive: {_dataset_name()}/{name}.pt")
            return torch.load(local, map_location="cpu", weights_only=False)
        if os.path.exists(drive_legacy):
            shutil.copy2(drive_legacy, local)
            print(f"  Wczytano z Drive (legacy): {name}.pt")
            return torch.load(local, map_location="cpu", weights_only=False)

    if os.path.exists(local):
        return torch.load(local, map_location="cpu", weights_only=False)

    # Legacy lokalnie
    legacy = os.path.join(cfg.CHECKPOINT_DIR, f"{name}.pt")
    if os.path.exists(legacy):
        return torch.load(legacy, map_location="cpu", weights_only=False)

    return None


def get_results_path(filename: str) -> str:
    return os.path.join(_ensure_dataset_dir(cfg.RESULTS_DIR), filename)
