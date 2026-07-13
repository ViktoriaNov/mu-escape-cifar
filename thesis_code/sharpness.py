"""
Pomiar perturbacyjnej ostrości minimum lokalnego.

Implementacja równań (2.8) i (4.1) z pracy:
  Ŝ_P(w, σ, K) = (1/K) Σ_k L(w + ε_k) − L(w),   ε_k ~ N(0, σ²I)

Interpretacja (rozdz. 2.3):
  - mała wartość → płaskie minimum (stabilne perturbacje wag)
  - duża wartość → ostre minimum (strata szybko rośnie po perturbacji)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CFG
from .training import evaluate


@torch.no_grad()
def _add_gaussian_noise_to_weights(model: nn.Module, sigma: float) -> None:
    """Dodaje ε ~ N(0, σ²) do każdej warstwy wag modelu."""
    for p in model.parameters():
        p.data.add_(torch.randn_like(p) * sigma)


def perturbation_sharpness(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    sigma: float | None = None,
    n_samples: int | None = None,
) -> Tuple[float, float]:
    """
    Oblicza perturbacyjną ostrość na podanym loaderze (zwykle walidacyjnym).

    Zwraca: (sharpness, strata_bazowa).
    Po pomiarze wagi są przywracane do stanu sprzed perturbacji.
    """
    sigma = sigma if sigma is not None else CFG["sharp_sigma"]
    n_samples = n_samples if n_samples is not None else CFG["sharp_n_samples"]

    base_loss, _ = evaluate(model, loader, device)
    original: List[torch.Tensor] = [p.data.clone() for p in model.parameters()]

    perturbed_losses: List[float] = []
    for _ in range(n_samples):
        _add_gaussian_noise_to_weights(model, sigma)
        pl, _ = evaluate(model, loader, device)
        perturbed_losses.append(pl)
        # Przywrócenie wag
        for p, orig in zip(model.parameters(), original):
            p.data.copy_(orig)

    sharpness = float(np.mean(perturbed_losses) - base_loss)
    return sharpness, base_loss


def measure_sharpness_mean(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_runs: int = 5,
) -> Tuple[float, float]:
    """Uśrednia K pomiarów ostrości — stabilniejszy wynik do tabel w rozdz. 6."""
    values = [
        perturbation_sharpness(model, loader, device)[0]
        for _ in range(n_runs)
    ]
    return float(np.mean(values)), float(np.std(values))
