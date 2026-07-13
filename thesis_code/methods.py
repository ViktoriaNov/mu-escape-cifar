"""
Implementacja metod porównawczych i MU Escape.

Metody (rozdz. 4):
  1. Early Stopping (ES)     — baseline, punkt startowy
  2. SGDR                    — cosine warm restarts (Loshchilov & Hutter, 2017)
  3. SWA                     — uśrednianie wag (Izmailov et al., 2018)
  4. Noise Injection         — perturbacja wag + fine-tuning
  5. MU Escape (proponowana) — gradient ascent + fine-tuning (alg. 1)
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CFG
from .diagnostics import format_escape_report, verify_escape
from .model import get_model, normalize_state_dict
from .sharpness import perturbation_sharpness
from .storage import save_checkpoint
from .training import CRITERION, evaluate, train_epoch


History = Dict[str, Any]


def _new_history() -> History:
    return defaultdict(list)


def run_early_stopping(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_classes: int,
    checkpoint_name: str = "es_best",
    verbose: bool = True,
) -> Tuple[nn.Module, History]:
    """
    Faza 1 — wczesne zatrzymanie z małym patience.

    Celowo małe p=3 (CFG['es_patience']) — model zatrzymuje się wcześniej,
    często w ostrzejszym minimum (hipoteza rozdz. 1 i 6).
    """
    model = get_model(num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CFG["es_lr"],
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 60], gamma=0.1
    )

    best_val_loss = float("inf")
    best_state: Optional[Dict] = None
    no_improve = 0
    history = _new_history()

    for epoch in range(1, CFG["es_max_epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if verbose and epoch % 5 == 0:
            print(
                f"  ES epoka {epoch:3d} | tr_acc={tr_acc:.3f} | "
                f"vl_acc={vl_acc:.3f} | vl_loss={vl_loss:.4f}"
            )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= CFG["es_patience"]:
            if verbose:
                print(f"\n  Early Stopping na epoce {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    _, test_acc = evaluate(model, test_loader, device)
    sharp, _ = perturbation_sharpness(model, val_loader, device)
    history["test_acc"] = test_acc
    history["sharpness"] = sharp
    save_checkpoint(best_state, checkpoint_name, dict(history))

    if verbose:
        print(f"  ES → test_acc={test_acc:.4f} | sharpness={sharp:.4f}")
    return model, dict(history)


def run_sgdr(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_classes: int,
    checkpoint_name: str = "sgdr_best",
    verbose: bool = True,
) -> Tuple[nn.Module, History]:
    """SGDR — cykliczne restarty η według cosinusa (rozdz. 3.2)."""
    model = get_model(num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=CFG["es_lr"], momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG["sgdr_T0"], T_mult=CFG["sgdr_Tmult"]
    )
    history = _new_history()

    for epoch in range(1, CFG["sgdr_epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        _, vl_acc = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_acc"].append(vl_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if verbose and epoch % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  SGDR epoka {epoch:3d} | tr_acc={tr_acc:.3f} | lr={lr:.5f}")

    _, test_acc = evaluate(model, test_loader, device)
    sharp, _ = perturbation_sharpness(model, val_loader, device)
    history["test_acc"] = test_acc
    history["sharpness"] = sharp
    save_checkpoint(model.state_dict(), checkpoint_name, dict(history))

    if verbose:
        print(f"  SGDR → test_acc={test_acc:.4f} | sharpness={sharp:.4f}")
    return model, dict(history)


def run_swa(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_classes: int,
    checkpoint_name: str = "swa_best",
    verbose: bool = True,
) -> Tuple[nn.Module, History]:
    """SWA — uśrednianie wag od epoki swa_start (rozdz. 3.2)."""
    model = get_model(num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=CFG["swa_lr"], momentum=0.9, weight_decay=5e-4
    )
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=CFG["swa_lr"], anneal_epochs=10
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["swa_start"]
    )
    history = _new_history()

    for epoch in range(1, CFG["swa_epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        _, vl_acc = evaluate(model, val_loader, device)

        if epoch >= CFG["swa_start"] and epoch % CFG["swa_freq"] == 0:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif epoch < CFG["swa_start"]:
            cosine.step()

        history["train_loss"].append(tr_loss)
        history["val_acc"].append(vl_acc)

        if verbose and epoch % 10 == 0:
            tag = " [SWA]" if epoch >= CFG["swa_start"] else ""
            print(f"  SWA epoka {epoch:3d} | tr_acc={tr_acc:.3f}{tag}")

    if verbose:
        print("  Przeliczanie BatchNorm dla modelu SWA...")
    swa_model.cpu()
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    swa_model.to(device)

    _, test_acc = evaluate(swa_model, test_loader, device)
    sharp, _ = perturbation_sharpness(swa_model, val_loader, device)
    history["test_acc"] = test_acc
    history["sharpness"] = sharp
    save_checkpoint(
        normalize_state_dict(swa_model.state_dict()),
        checkpoint_name,
        dict(history),
    )

    if verbose:
        print(f"  SWA → test_acc={test_acc:.4f} | sharpness={sharp:.4f}")
    return swa_model, dict(history)


def run_noise_injection(
    starting_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    noise_std: float | None = None,
    checkpoint_name: str = "ni_best",
    verbose: bool = True,
) -> Tuple[nn.Module, History]:
    """
    Noise Injection — start z modelu ES, szum na wagach, potem fine-tuning.

    Porównywalny z MU Escape: ten sam punkt startowy i ta sama faza FT,
    różni się tylko sposób destabilizacji (losowy szum vs gradient ascent).
    """
    noise_std = noise_std if noise_std is not None else CFG["noise_std"]
    model = copy.deepcopy(starting_model).to(device)
    history = _new_history()

    if verbose:
        print(f"  Noise Injection: szum std={noise_std}")
    with torch.no_grad():
        for p in model.parameters():
            p.data.add_(torch.randn_like(p) * noise_std)

    sharp_before, _ = perturbation_sharpness(model, val_loader, device)
    if verbose:
        print(f"  Sharpness po szumie: {sharp_before:.4f}")

    model, history = _fine_tune(
        model, train_loader, val_loader, test_loader, device,
        history, checkpoint_name, verbose,
    )
    return model, history


def run_mu_escape(
    starting_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    ul_loader: DataLoader,
    device: str,
    checkpoint_name: str = "mu_best",
    verbose: bool = True,
) -> Tuple[nn.Module, History]:
    """
    MU Escape — proponowana metoda (algorytm 1, rozdz. 4).

    Etap B: gradient ascent na D_f — maksymalizacja straty (odwrotny znak gradientu)
    Etap C: fine-tuning na pełnym D_train — powrót do niskiej straty, nowe minimum
    """
    model = copy.deepcopy(starting_model).to(device)
    history = _new_history()

    # Kopia wag ES — do weryfikacji escape (por. diagnostics.py)
    es_snapshot = copy.deepcopy(starting_model).to(device)

    # ── Etap B: Gradient Ascent ─────────────────────────────────────────────
    if verbose:
        print("  === MU Escape: Gradient Ascent ===")
    ul_optimizer = torch.optim.SGD(
        model.parameters(), lr=CFG["ul_lr"], momentum=0.9
    )

    _, val_acc_before = evaluate(model, val_loader, device)
    sharp_before, _ = perturbation_sharpness(model, val_loader, device)
    if verbose:
        print(
            f"  Przed GA: vl_acc={val_acc_before:.4f} | "
            f"sharpness={sharp_before:.4f}"
        )

    ul_iter = iter(ul_loader)
    model.train()
    for step in range(CFG["ul_steps"]):
        try:
            x, y = next(ul_iter)
        except StopIteration:
            ul_iter = iter(ul_loader)
            x, y = next(ul_iter)

        x, y = x.to(device), y.to(device)
        ul_optimizer.zero_grad()
        loss = CRITERION(model(x), y)
        # Gradient ASCENT: minimalizujemy (-loss) ≡ maksymalizujemy loss
        (-loss).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=CFG["ul_clip_norm"]
        )
        ul_optimizer.step()

        if step % 5 == 0:
            _, va = evaluate(model, val_loader, device)
            history["ul_val_acc"].append(va)
            if verbose:
                print(f"    krok {step:3d} | loss={loss.item():.4f} | vl_acc={va:.4f}")
            if va < CFG["ul_val_threshold"]:
                if verbose:
                    print("    Przerwanie GA — vl_acc poniżej progu")
                break

    sharp_after_ga, _ = perturbation_sharpness(model, val_loader, device)
    history["sharp_before"] = sharp_before
    history["sharp_after_ul"] = sharp_after_ga
    if verbose:
        print(f"  Po GA: sharpness={sharp_after_ga:.4f} (było {sharp_before:.4f})")

    # Snapshot po GA (przed fine-tuningiem)
    ga_snapshot = copy.deepcopy(model).to(device)

    # ── Etap C: Fine-tuning ─────────────────────────────────────────────────
    if verbose:
        print("  === MU Escape: Fine-tuning ===")
    model, history = _fine_tune(
        model, train_loader, val_loader, test_loader, device,
        history, checkpoint_name, verbose,
    )
    history["sharp_before"] = sharp_before
    history["sharp_after_ul"] = sharp_after_ga

    escape_report = verify_escape(
        es_snapshot,
        ga_snapshot,
        model,
        val_loader,
        device,
        sharp_before,
        sharp_after_ga,
        history["sharpness"],
        compute_hessian=False,
    )
    history["escape_diagnostics"] = escape_report
    if verbose:
        print(format_escape_report(escape_report))

    return model, dict(history)


def _fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    history: History,
    checkpoint_name: str,
    verbose: bool,
) -> Tuple[nn.Module, History]:
    """Wspólna faza douczania dla MU Escape i Noise Injection."""
    ft_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CFG["ft_lr"],
        momentum=0.9,
        weight_decay=5e-4,
    )
    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ft_optimizer, T_max=CFG["ft_epochs"]
    )

    best_val = 0.0
    best_state: Optional[Dict] = None

    for epoch in range(1, CFG["ft_epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, ft_optimizer, device)
        _, vl_acc = evaluate(model, val_loader, device)
        ft_scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val:
            best_val = vl_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose and epoch % 5 == 0:
            print(f"    FT epoka {epoch:3d} | tr_acc={tr_acc:.3f} | vl_acc={vl_acc:.3f}")

    assert best_state is not None
    model.load_state_dict(best_state)

    _, test_acc = evaluate(model, test_loader, device)
    sharp_final, _ = perturbation_sharpness(model, val_loader, device)
    history["test_acc"] = test_acc
    history["sharpness"] = sharp_final
    save_checkpoint(best_state, checkpoint_name, dict(history))

    if verbose:
        print(f"  Po FT → test_acc={test_acc:.4f} | sharpness={sharp_final:.4f}")
    return model, dict(history)
