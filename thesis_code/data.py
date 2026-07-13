"""
Wczytywanie i podział zbiorów CIFAR.

Odpowiada sekcji 2.6 pracy:
  - 50 000 obrazów treningowych → 45 000 train + 5 000 val (10%)
  - augmentacje: RandomCrop(32, padding=4), RandomHorizontalFlip
  - normalizacja statystykami CIFAR
  - podzbiór ul_loader: α_f · |D_train| — dane do fazy gradient ascent
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from .config import CFG, DATA_DIR


# Statystyki normalizacji CIFAR (rozdz. 2.6)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def get_loaders(
    dataset_name: str | None = None,
    seed: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
    """
    Zwraca: train_loader, val_loader, test_loader, ul_loader, num_classes.

    ul_loader — losowy podzbiór treningowy do gradient ascent (MU Escape).
    Podział train/val jest deterministyczny względem `seed`.
    """
    dataset_name = dataset_name or CFG["dataset"]
    seed = seed if seed is not None else CFG["seed"]

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    ds_cls = (
        torchvision.datasets.CIFAR10
        if dataset_name == "CIFAR10"
        else torchvision.datasets.CIFAR100
    )
    num_classes = 10 if dataset_name == "CIFAR10" else 100

    train_ds = ds_cls(root=DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = ds_cls(root=DATA_DIR, train=False, download=True, transform=test_tf)

    # Deterministyczny podział 90/10 train/val
    n = len(train_ds)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_n = int(0.1 * n)
    val_idx = perm[:val_n].tolist()
    train_idx = perm[val_n:].tolist()

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(train_ds, val_idx),
        batch_size=256,
        shuffle=False,
        num_workers=CFG["num_workers"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=CFG["num_workers"],
    )

    # Podzbiór D_f do gradient ascent — pierwsze α_f · |train| indeksów
    # (po permutacji train_idx; w praktyce losowa próbka z train_idx)
    ul_n = int(CFG["ul_fraction"] * len(train_idx))
    ul_subset_idx = rng.choice(train_idx, size=ul_n, replace=False).tolist()
    ul_loader = DataLoader(
        Subset(train_ds, ul_subset_idx),
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
    )

    print(
        f"Zbiór: {dataset_name} | train: {len(train_idx)} | "
        f"val: {val_n} | test: {len(test_ds)} | ul: {ul_n} | klas: {num_classes}"
    )
    return train_loader, val_loader, test_loader, ul_loader, num_classes
