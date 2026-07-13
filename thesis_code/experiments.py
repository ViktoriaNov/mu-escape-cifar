"""
Orkiestracja eksperymentów — multi-seed, tabele, wykresy (rozdz. 6).

Uruchomienie:
  python -m thesis_code.experiments          # pełny protokół (długo!)
  python -m thesis_code.experiments --quick # jedno ziarno, szybki test
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

import pandas as pd

from .config import CFG, get_device, set_seed
from .data import get_loaders
from .methods import (
    run_early_stopping,
    run_mu_escape,
    run_noise_injection,
    run_sgdr,
    run_swa,
)
from .model import get_model, load_model_weights
from .training import evaluate
from .sharpness import measure_sharpness_mean, perturbation_sharpness
from .storage import get_results_path, load_checkpoint, save_checkpoint, save_to_drive, setup_paths
from .visualize import build_summary_table, plot_patience_diagnostic, save_and_show_results

# Kolejność metod i prefiksy plików checkpointów
METHOD_PIPELINE = [
    ("Early Stopping", "es", "run_es"),
    ("SGDR", "sgdr", "run_sgdr"),
    ("SWA", "swa", "run_swa"),
    ("Noise Injection", "ni", "run_ni"),
    ("MU Escape", "mu", "run_mu"),
]


def _result_from_checkpoint(
    method: str,
    seed: int,
    prefix: str,
    device: str,
    val_loader,
    test_loader,
    quiet: bool = False,
) -> Dict[str, Any] | None:
    """Wczytuje wynik z checkpointu jeśli istnieje (resume / inny account)."""
    ckpt = load_checkpoint(f"{prefix}_seed{seed}")
    if ckpt is None:
        return None

    num_classes = 10 if CFG["dataset"] == "CIFAR10" else 100
    model = get_model(num_classes).to(device)
    load_model_weights(model, ckpt["state_dict"])
    history = dict(ckpt.get("history", {}))
    needs_save = False

    if history.get("sharpness") is not None:
        sh_m = float(history["sharpness"])
        sh_s = 0.0
    else:
        if not quiet:
            print(
                f"  ↩ {method} seed={seed} — szybki pomiar ostrości (~1–2 min)..."
            )
        sh_m, _ = perturbation_sharpness(
            model, val_loader, device, n_samples=5,
        )
        sh_s = 0.0
        history["sharpness"] = sh_m
        needs_save = True

    acc = float(history.get("test_acc") or 0.0)
    if acc <= 0.0:
        if not quiet:
            print(f"  ↩ {method} seed={seed} — liczę test_acc (~10 s)...")
        _, acc = evaluate(model, test_loader, device)
        history["test_acc"] = acc
        needs_save = True

    if needs_save:
        save_checkpoint(ckpt["state_dict"], f"{prefix}_seed{seed}", history)

    if not quiet:
        print(
            f"  ↩ pominięto (checkpoint): {method} seed={seed} | "
            f"test_acc={acc:.4f} | sharpness={sh_m:.4f}"
        )

    return {
        "seed": seed,
        "test_acc": acc,
        "sharpness_mean": sh_m,
        "sharpness_std": sh_s,
        "history": history,
        "_model": model,
    }


METHOD_NAMES = [name for name, _, _ in METHOD_PIPELINE]
_CHECKPOINT_PREFIX = {name: prefix for name, prefix, _ in METHOD_PIPELINE}


def empty_results() -> Dict[str, List[Dict[str, Any]]]:
    """Pusty słownik wyników — do notatnika (jedna metoda na komórkę)."""
    return {name: [] for name in METHOD_NAMES}


def append_result(
    all_results: Dict[str, List[Dict[str, Any]]],
    method: str,
    result: Dict[str, Any] | None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Dopisuje wynik metody; nadpisuje jeśli to samo ziarno już jest."""
    if result is None:
        return all_results
    lst = all_results.setdefault(method, [])
    for i, row in enumerate(lst):
        if row["seed"] == result["seed"]:
            lst[i] = result
            return all_results
    lst.append(result)
    return all_results


def _load_es_model(seed: int, device: str, num_classes: int):
    """Model ES z checkpointu — wymagany dla Noise Injection i MU Escape."""
    ckpt = load_checkpoint(f"es_seed{seed}")
    if ckpt is None:
        raise RuntimeError(
            f"Brak checkpointu es_seed{seed}. "
            "Najpierw uruchom komórkę Early Stopping."
        )
    model = get_model(num_classes).to(device)
    load_model_weights(model, ckpt["state_dict"])
    return model


def run_method(
    method: str,
    seed: int,
    device: str,
    resume: bool = True,
    verbose: bool = True,
) -> Dict[str, Any] | None:
    """
    Jedna metoda, jedno ziarno — do osobnych komórek w notatniku.

    Zwraca słownik wyniku lub None, gdy resume=True i checkpoint już istnieje
    (wtedy wynik jest wczytany z dysku i też zwracany).
    """
    if method not in METHOD_NAMES:
        raise ValueError(f"Nieznana metoda: {method!r}. Dostępne: {METHOD_NAMES}")

    set_seed(seed)
    train_loader, val_loader, test_loader, ul_loader, num_classes = get_loaders(
        seed=seed
    )
    prefix = _CHECKPOINT_PREFIX[method]

    if resume:
        cached = _result_from_checkpoint(
            method, seed, prefix, device, val_loader, test_loader,
        )
        if cached:
            cached.pop("_model", None)
            return cached

    print(f"\n--- Ziarno {seed}: {method} ---")

    if method == "Early Stopping":
        model, hist = run_early_stopping(
            train_loader, val_loader, test_loader, device, num_classes,
            checkpoint_name=f"es_seed{seed}", verbose=verbose,
        )
    elif method == "SGDR":
        model, hist = run_sgdr(
            train_loader, val_loader, test_loader, device, num_classes,
            checkpoint_name=f"sgdr_seed{seed}", verbose=verbose,
        )
    elif method == "SWA":
        model, hist = run_swa(
            train_loader, val_loader, test_loader, device, num_classes,
            checkpoint_name=f"swa_seed{seed}", verbose=verbose,
        )
    elif method == "Noise Injection":
        es_model = _load_es_model(seed, device, num_classes)
        model, hist = run_noise_injection(
            es_model, train_loader, val_loader, test_loader, device,
            checkpoint_name=f"ni_seed{seed}", verbose=verbose,
        )
    elif method == "MU Escape":
        es_model = _load_es_model(seed, device, num_classes)
        model, hist = run_mu_escape(
            es_model, train_loader, val_loader, test_loader, ul_loader, device,
            checkpoint_name=f"mu_seed{seed}", verbose=verbose,
        )
    else:
        raise ValueError(method)

    sh_m, sh_s = measure_sharpness_mean(model, val_loader, device)
    return {
        "seed": seed,
        "test_acc": hist["test_acc"],
        "sharpness_mean": sh_m,
        "sharpness_std": sh_s,
        "history": hist,
    }


def run_all_methods_for_seed(
    seed: int,
    device: str,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Pełny protokół dla jednego ziarna: ES → SGDR → SWA → NI → MU Escape."""
    results = empty_results()
    for method in METHOD_NAMES:
        row = run_method(method, seed, device, resume=resume, verbose=verbose)
        append_result(results, method, row)
    return results


def print_resume_status(seeds: List[int] | None = None) -> None:
    """Pokazuje które checkpointy już są na Drive / dysku."""
    seeds = seeds or CFG["seeds"]
    prefixes = ["es", "sgdr", "swa", "ni", "mu"]
    names = ["ES", "SGDR", "SWA", "NI", "MU"]
    print(f"Dataset: {CFG['dataset']}")
    print(f"{'Seed':>6}  " + "  ".join(f"{n:>5}" for n in names))
    for seed in seeds:
        row = []
        for p in prefixes:
            ok = load_checkpoint(f"{p}_seed{seed}") is not None
            row.append("  ✓  " if ok else "  ·  ")
        print(f"{seed:>6}  " + "".join(row))


def merge_results(
    acc: Dict[str, List],
    new: Dict[str, List],
) -> Dict[str, List]:
    for k, v in new.items():
        acc.setdefault(k, []).extend(v)
    return acc


def run_patience_diagnostic(
    device: str,
    num_classes: int,
    seed: int | None = None,
    patience_values: list[int] | None = None,
    sigma_sweep: list[float] | None = None,
    verbose: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    """
    Eksperyment z rozdz. 6 — wpływ patience na ostrość minimum.

    Uruchamia ES z patience ∈ {3, 5, 7, 15} (domyślnie), zapisuje:
      - results/{dataset}/diagnostyka_patience.csv
      - results/{dataset}/diagnostyka_sharpness.png

    Opcjonalnie: sweep σ dla patience=3 i patience=7 (modele w checkpointach
    es_p3, es_p7) — wykres trójpanelowy w diagnostyka_sharpness.png.
    """
    from .model import get_model, load_model_weights
    from .sharpness import perturbation_sharpness

    patience_values = patience_values or [3, 5, 7, 15]
    sigma_sweep = sigma_sweep or [0.005, 0.01, 0.015, 0.02, 0.03]
    seed = seed if seed is not None else CFG["seed"]
    original_patience = CFG["es_patience"]

    rows = []
    sigma_models: dict[int, Any] = {}

    for patience in patience_values:
        CFG["es_patience"] = patience
        set_seed(seed)
        train_loader, val_loader, test_loader, _, _ = get_loaders(seed=seed)
        if verbose:
            print(f"\n--- Patience diagnostic: p={patience}, seed={seed} ---")
        _, h = run_early_stopping(
            train_loader, val_loader, test_loader, device, num_classes,
            checkpoint_name=f"es_p{patience}", verbose=verbose,
        )
        rows.append({
            "patience": patience,
            "seed": seed,
            "epoka_stop": len(h["val_acc"]),
            "test_acc": h["test_acc"],
            "sharpness": h["sharpness"],
        })
        if patience in (3, 7):
            ckpt = load_checkpoint(f"es_p{patience}")
            if ckpt:
                model = get_model(num_classes).to(device)
                load_model_weights(model, ckpt["state_dict"])
                sigma_models[patience] = model

    CFG["es_patience"] = original_patience
    df = pd.DataFrame(rows)
    csv_path = get_results_path("diagnostyka_patience.csv")
    df.to_csv(csv_path, index=False)
    save_to_drive(csv_path, f"results/{CFG['dataset']}/diagnostyka_patience.csv")

    sigma_curves: dict[str, list[float]] | None = None
    if len(sigma_models) >= 2:
        sigma_curves = {}
        for p, model in sorted(sigma_models.items()):
            values = [
                perturbation_sharpness(model, val_loader, device, sigma=s)[0]
                for s in sigma_sweep
            ]
            sigma_curves[f"patience={p}"] = values

    png_path = get_results_path("diagnostyka_sharpness.png")
    plot_patience_diagnostic(
        df,
        sigma_curves=sigma_curves,
        sigmas=sigma_sweep if sigma_curves else None,
        out_path=png_path,
        show=show,
    )

    if verbose:
        print("\n" + df.to_string(index=False))
        print(f"\nZapisano: {csv_path}")
        print(f"         {png_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksperymenty MU Escape")
    parser.add_argument(
        "--quick", action="store_true",
        help="Jedno ziarno, bez multi-seed (szybki test)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Pomiń metody, które mają już checkpoint na Drive/dysku",
    )
    parser.add_argument(
        "--patience", action="store_true",
        help="Tylko eksperyment diagnostyczny patience (rozdz. 6)",
    )
    parser.add_argument(
        "--dataset", choices=["CIFAR10", "CIFAR100"], default=None,
        help="Zbiór danych (domyślnie z config.py)",
    )
    args = parser.parse_args()

    setup_paths()
    if args.dataset:
        CFG["dataset"] = args.dataset
    device = get_device()
    print(f"Urządzenie: {device} | dataset: {CFG['dataset']}")

    if args.patience:
        num_classes = 10 if CFG["dataset"] == "CIFAR10" else 100
        run_patience_diagnostic(device, num_classes, verbose=True, show=False)
        return

    seeds = [CFG["seed"]] if args.quick else CFG["seeds"]
    if args.resume:
        print_resume_status(seeds)

    all_results: Dict[str, List] = {}
    t0 = time.time()

    for seed in seeds:
        partial = run_all_methods_for_seed(
            seed, device, resume=args.resume,
        )
        all_results = merge_results(all_results, partial)

    df = build_summary_table(all_results)
    save_and_show_results(all_results, seeds[-1], show=False)

    print("\n" + "=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)
    print(f"\nPliki: results/{CFG['dataset']}/")

    print(f"\nCzas: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
