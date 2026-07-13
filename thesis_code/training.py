"""
Podstawowe operacje treningowe — SGD w jednej epoce i ewaluacja.

Odpowiada równaniu (2.2) w pracy: aktualizacja wag przez mini-batch SGD
z entropią krzyżową jako funkcją straty (rozdz. 2.1).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Wspólna funkcja straty dla wszystkich eksperymentów
CRITERION = nn.CrossEntropyLoss()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """
    Jedna epoka uczenia: minimalizacja straty na mini-paczkach.

    Zwraca: (średnia strata treningowa, dokładność treningowa).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = CRITERION(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """
    Ewaluacja bez aktualizacji wag.

    Zwraca: (średnia strata, dokładność) — używane na val i test.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += CRITERION(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total
