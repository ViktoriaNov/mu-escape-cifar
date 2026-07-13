"""
Konfiguracja eksperymentów — jedno miejsce na wszystkie hiperparametry.

Wartości zgodne z tabelą w rozdziale 4 (metodologia) pracy dyplomowej.
Zmiana tutaj automatycznie propaguje się do wszystkich eksperymentów.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import torch

# ── Hiperparametry (odpowiadają tab. parametrów w rozdz. 4) ─────────────────

CFG: Dict[str, Any] = {
    # Zbiór danych: 'CIFAR10' lub 'CIFAR100' (rozdz. 2.6)
    "dataset": "CIFAR10",
    "batch_size": 128,
    "num_workers": 2,
    "seed": 42,
    # Early Stopping — baseline i punkt startowy MU Escape / Noise Injection
    # Mały patience celowo prowadzi do ostrszego minimum (rozdz. 2.4, 4.2)
    "es_patience": 3,
    "es_max_epochs": 80,
    "es_lr": 0.1,
    # Fine-tuning po destabilizacji (etap C MU Escape, Noise Injection)
    "ft_epochs": 30,
    "ft_lr": 0.01,
    # MU Escape — faza gradient ascent (rozdz. 4.1, alg. 1)
    "ul_steps": 20,           # T_GA — liczba kroków gradientowych
    "ul_lr": 5e-5,            # η_GA — mały lr zapobiega NaN
    "ul_fraction": 0.05,      # α_f — 5% danych treningowych
    "ul_clip_norm": 0.5,      # ρ_clip — obcinanie gradientu
    "ul_val_threshold": 0.20, # τ_min — próg wczesnego przerwania GA
    # SGDR (Loshchilov & Hutter, 2017)
    "sgdr_T0": 10,
    "sgdr_Tmult": 2,
    "sgdr_epochs": 80,
    # SWA (Izmailov et al., 2018)
    "swa_start": 50,
    "swa_freq": 5,
    "swa_lr": 0.05,
    "swa_epochs": 80,
    # Noise Injection — σ perturbacji wag (rozdz. 4.2)
    "noise_std": 0.005,
    # Perturbacyjna ostrość — równ. (2.8) i (4.1) w pracy
    "sharp_sigma": 0.01,
    "sharp_n_samples": 20,
    # Multi-seed (rozdz. 6) — powtarzalność wyników mean ± std
    "seeds": [42, 123, 777],
}

# Ścieżki projektu (nadpisywane w storage.setup_paths)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def get_device() -> str:
    """GPU jeśli dostępne (Colab T4 / lokalna karta NVIDIA), inaczej CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Ustalenie ziarna dla powtarzalności (PyTorch, NumPy, cuDNN)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
