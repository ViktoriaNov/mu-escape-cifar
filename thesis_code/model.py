"""
Architektura ResNet-18 dostosowana do obrazów CIFAR 32×32.

Standardowy ResNet-18 (torchvision) jest projektowany pod ImageNet 224×224.
Modyfikacje zgodne z rozdz. 2.6 i literaturą (Izmailov et al., Foret et al.):
  - conv1: 7×7, stride 2  →  3×3, stride 1, padding 1
  - maxpool: usunięty (zastąpiony Identity)
  - fc: 512 → num_classes (10 lub 100)

Model ma ~11,2 mln parametrów.
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
from torchvision.models import resnet18


def normalize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ujednolica zapis wag — m.in. SWA (AveragedModel) ma klucze ``module.*``.
    """
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
        if k != "n_averaged"
    }


def load_model_weights(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    """Wczytuje wagi do zwykłego ResNet-18 (obsługa starych checkpointów SWA)."""
    model.load_state_dict(normalize_state_dict(state_dict))


def get_model(num_classes: int = 10) -> nn.Module:
    """Tworzy świeży ResNet-18 dla CIFAR (bez pretrenowanych wag)."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Liczba uczalnych parametrów (do rozdz. 5 — środowisko)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
