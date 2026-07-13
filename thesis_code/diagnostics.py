"""
Diagnostyka geometrii optimum — czy punkt jest blisko minimum i czy nastąpił escape.

Teoria (rozdz. 2):
  - Punkt krytyczny:  ||∇L(w)|| ≈ 0
  - Minimum lokalne:  ∇L = 0  ORAZ  λ_max(Hessian) > 0
  - Escape z minimum: wagi i strata się zmieniają; po GA strata rośnie

W praktyce dla dużych sieci NIE liczymy pełnego Hessianu (11 mln parametrów).
Stosujemy tanie proxy, akceptowane w literaturze:
  1. norma gradientu na zbiorze walidacyjnym
  2. perturbacyjna ostrość (już w sharpness.py)
  3. odległość L2 między wektorami wag ||w - w_0||
  4. strata walidacyjna przed / po GA / po FT
  5. (opcjonalnie) największa wartość własna Hessianu — power iteration
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .training import CRITERION, evaluate


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Spłaszcza wszystkie wagi do jednego wektora (do pomiaru odległości)."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def weight_l2_distance(model_a: nn.Module, model_b: nn.Module) -> float:
    """
    ||w_a - w_b||_2 — jeśli ≈ 0, to ten sam punkt w przestrzeni parametrów.
    Po udanym escape po GA powinno być wyraźnie > 0.
    """
    wa = flatten_params(model_a)
    wb = flatten_params(model_b)
    return float(torch.norm(wa - wb, p=2).item())


def weight_cosine_similarity(model_a: nn.Module, model_b: nn.Module) -> float:
    """
    Podobieństwo kierunku wag w [-1, 1].
    cos ≈ 1 → prawie ten sam kierunek; po escape zwykle spada.
    """
    wa = flatten_params(model_a)
    wb = flatten_params(model_b)
    denom = torch.norm(wa) * torch.norm(wb)
    if denom < 1e-12:
        return 1.0
    return float(torch.dot(wa, wb).item() / denom.item())


def gradient_norm(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int | None = None,
) -> float:
    """
    Średnia norma L2 gradientu po mini-paczkach (proxy bliskości punktu krytycznego).

    Interpretacja (nie jest twierdzeniem o minimum!):
      - ||g|| małe  → blisko stacjonarności (SGD praktycznie się zatrzymał)
      - ||g|| duże  → wciąż aktywna optymalizacja lub wyjście z basenu (po GA)

    Uwaga: mały gradient NIE dowodzi minimum lokalnego — może to być płaski
    plateau lub punkt siodłowy (rozdz. 2.2).
    """
    was_training = model.training
    model.eval()
    norms: List[float] = []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = CRITERION(model(x), y)
        loss.backward()
        sq = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
        norms.append(sq ** 0.5)
    model.train(was_training)
    return float(sum(norms) / len(norms)) if norms else 0.0


def top_hessian_eigenvalue(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_power_iter: int = 20,
    max_batches: int = 1,
) -> float:
    """
    Szacunek λ_max(Hessian) metodą potęgową (Hessian-vector product).

    λ_max > 0 → lokalne nachylenie w górę we wszystkich kierunkach (wskazuje na minimum).
    λ_max < 0 lub mieszane → możliwy punkt siodłowy.

    Kosztowne — domyślnie 1 mini-paczka walidacyjna, 20 iteracji.
    Dla ResNet-18 wystarczy jako dodatkowy dowód w pracy.
    """
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    # Losowy wektor jednostkowy v w przestrzeni wag
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    v_norm = torch.sqrt(sum((vi ** 2).sum() for vi in v))
    v = [vi / (v_norm + 1e-12) for vi in v]

    def hvp(vec: List[torch.Tensor]) -> List[torch.Tensor]:
        model.zero_grad()
        loss = CRITERION(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * vi).sum() for g, vi in zip(grads, vec))
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        return list(hv)

    eigenvalue = 0.0
    for _ in range(n_power_iter):
        Hv = hvp(v)
        eigenvalue = sum((vi * hvi).sum().item() for vi, hvi in zip(v, Hv))
        h_norm = torch.sqrt(sum((hvi ** 2).sum() for hvi in Hv))
        if h_norm < 1e-12:
            break
        v = [hvi / h_norm for hvi in Hv]

    return float(eigenvalue)


def diagnose_point(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    label: str = "",
    compute_hessian: bool = False,
) -> Dict[str, Any]:
    """Pełna diagnostyka jednego punktu w przestrzeni parametrów."""
    val_loss, val_acc = evaluate(model, loader, device)
    grad_norm = gradient_norm(model, loader, device, max_batches=50)

    out: Dict[str, Any] = {
        "label": label,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "grad_norm": grad_norm,
    }
    if compute_hessian:
        out["lambda_max_hessian"] = top_hessian_eigenvalue(
            model, loader, device, n_power_iter=15, max_batches=1
        )
    return out


def verify_escape(
    model_before: nn.Module,
    model_after_ga: nn.Module,
    model_after_ft: nn.Module,
    val_loader: DataLoader,
    device: str,
    sharp_before: float,
    sharp_after_ga: float,
    sharp_after_ft: float,
    compute_hessian: bool = False,
) -> Dict[str, Any]:
    """
    Weryfikacja hipotezy escape dla MU Escape (rozdz. 4, 6).

    Kryteria SUKCESU escape (empiryczne, nie formalny dowód):
      A. Po GA: val_loss wyższa niż przed GA  (wyszliśmy z doliny)
      B. Po GA: sharpness wyższa               (bardziej zakrzywiony region)
      C. Po GA: ||w - w_0|| > 0                (wagi się przesunęły)
      D. Po FT: val_acc lepsza niż ES baseline (nowe, lepsze minimum)
      E. Po FT: ||w* - w_0|| > 0 i cos < 1    (nie wróciliśmy do tego samego punktu)
    """
    d0 = diagnose_point(model_before, val_loader, device, "przed_GA", compute_hessian)
    d1 = diagnose_point(model_after_ga, val_loader, device, "po_GA", compute_hessian)
    d2 = diagnose_point(model_after_ft, val_loader, device, "po_FT", compute_hessian)

    dist_ga = weight_l2_distance(model_before, model_after_ga)
    dist_ft = weight_l2_distance(model_before, model_after_ft)
    cos_ga = weight_cosine_similarity(model_before, model_after_ga)
    cos_ft = weight_cosine_similarity(model_before, model_after_ft)

    checks = {
        "A_loss_wzrosła_po_GA": d1["val_loss"] > d0["val_loss"],
        "B_sharpness_wzrosła_po_GA": sharp_after_ga > sharp_before,
        "C_wagi_się_przesunęły_po_GA": dist_ga > 1e-6,
        "D_acc_lepsza_po_FT_niż_ES": d2["val_acc"] > d0["val_acc"],
        "E_inne_minimum_po_FT": dist_ft > 1e-6 and cos_ft < 0.9999,
        "F_strata_po_FT_niższa_niż_po_GA": d2["val_loss"] < d1["val_loss"],
    }
    n_pass = sum(checks.values())

    return {
        "przed_GA": d0,
        "po_GA": d1,
        "po_FT": d2,
        "sharpness": {
            "przed": sharp_before,
            "po_GA": sharp_after_ga,
            "po_FT": sharp_after_ft,
        },
        "weight_distance_L2": {"po_GA": dist_ga, "po_FT": dist_ft},
        "weight_cosine": {"po_GA": cos_ga, "po_FT": cos_ft},
        "checks": checks,
        "checks_passed": n_pass,
        "checks_total": len(checks),
        "escape_verified": n_pass >= 4,  # próg: 4/6 kryteriów
    }


def format_escape_report(report: Dict[str, Any]) -> str:
    """Czytelny raport tekstowy do notatnika / logów."""
    lines = [
        "=" * 60,
        "RAPORT WERYFIKACJI ESCAPE",
        "=" * 60,
    ]
    for stage in ("przed_GA", "po_GA", "po_FT"):
        d = report[stage]
        lines.append(
            f"\n[{d['label']}]  val_loss={d['val_loss']:.4f}  "
            f"val_acc={d['val_acc']:.4f}  ||grad||={d['grad_norm']:.6f}"
        )
        if "lambda_max_hessian" in d:
            lines.append(f"  λ_max(Hessian) ≈ {d['lambda_max_hessian']:.4f}")

    s = report["sharpness"]
    lines.append(
        f"\nOstrość: {s['przed']:.4f} → {s['po_GA']:.4f} → {s['po_FT']:.4f}"
    )
    wd = report["weight_distance_L2"]
    lines.append(f"Odległość wag od ES: po GA={wd['po_GA']:.2f}, po FT={wd['po_FT']:.2f}")

    lines.append("\nKryteria:")
    for name, ok in report["checks"].items():
        lines.append(f"  {'✓' if ok else '✗'}  {name}")

    lines.append(
        f"\nWynik: {report['checks_passed']}/{report['checks_total']} — "
        f"escape {'POTWIERDZONY' if report['escape_verified'] else 'NIEPEWNY'}"
    )
    lines.append("=" * 60)
    return "\n".join(lines)
