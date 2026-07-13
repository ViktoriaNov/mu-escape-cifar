"""
Wykresy i podgląd wyników — zapis do results/ + wyświetlenie w notatniku.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CFG
from .storage import get_results_path, save_to_drive

# Styl wykresów do pracy (białe tło, bez przezroczystości)
THESIS_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.facecolor": "white",
    "legend.edgecolor": "#b0b0b0",
    "legend.framealpha": 1.0,
}
plt.rcParams.update(THESIS_RC)

METHOD_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
METHOD_SHORT = ["ES", "SGDR", "SWA", "NI", "MU"]
METHOD_COLOR_MAP = {
    "Early Stopping": METHOD_COLORS[0],
    "SGDR": METHOD_COLORS[1],
    "SWA": METHOD_COLORS[2],
    "Noise Injection": METHOD_COLORS[3],
    "MU Escape": METHOD_COLORS[4],
}


def _method_color(name: str) -> str:
    return METHOD_COLOR_MAP.get(name, METHOD_COLORS[0])


def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    save_to_drive(path, f"results/{CFG['dataset']}/{os.path.basename(path)}")


def build_summary_table(multi_seed: Dict[str, List]) -> pd.DataFrame:
    """Tabela mean ± std."""
    rows = []
    for method, runs in multi_seed.items():
        if not runs:
            continue
        accs = [r["test_acc"] for r in runs]
        sharps = [r["sharpness_mean"] for r in runs]
        rows.append({
            "Metoda": method,
            "Dokladnosc": f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
            "Sharpness": f"{np.mean(sharps):.4f} ± {np.std(sharps):.4f}",
            "Ziarna": len(runs),
        })
    return pd.DataFrame(rows)


def plot_patience_diagnostic(
    df: pd.DataFrame,
    sigma_curves: Optional[Dict[str, List[float]]] = None,
    sigmas: Optional[List[float]] = None,
    out_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Wykres diagnostyki patience (rozdz. 6) — panele zapisane osobno (układ pionowy w LaTeX).
    """
    ds = CFG["dataset"]
    acc_path = get_results_path("diagnostyka_patience_acc.png")
    sharp_path = get_results_path("diagnostyka_patience_sharp.png")
    panel_paths: List[str] = []

    if sigma_curves and sigmas:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        for label, values in sigma_curves.items():
            axes[0].plot(sigmas, values, "o-", label=label, linewidth=2)
        axes[0].set_xlabel(r"$\sigma$ perturbacji")
        axes[0].set_ylabel(r"Ostrość $\hat{S}_P$")
        axes[0].set_title("Czułość estymatora ostrości")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        panel_paths.append(acc_path)  # placeholder; sigma panel optional

        x = np.arange(len(df))
        width = 0.35
        axes[1].bar(x - width / 2, df["test_acc"], width, label="test acc", color="#3498db")
        ax1b = axes[1].twinx()
        ax1b.plot(x, df["epoka_stop"], "s--", color="#e74c3c", label="epoka stop")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(p) for p in df["patience"]])
        axes[1].set_xlabel("patience $p$")
        axes[1].set_ylabel("Dokładność testowa")
        ax1b.set_ylabel("Epoka zatrzymania")
        axes[1].set_title("Wpływ patience na jakość i długość uczenia")
        axes[1].grid(alpha=0.3, axis="y")

        axes[2].bar(
            [str(p) for p in df["patience"]],
            df["sharpness"],
            color="#9b59b6",
            edgecolor="gray",
        )
        axes[2].set_xlabel("patience $p$")
        axes[2].set_ylabel(r"Ostrość ($\sigma$=" + f"{CFG['sharp_sigma']})")
        axes[2].set_title("Ostrość minimum po ES")
        for i, v in enumerate(df["sharpness"]):
            axes[2].text(i, v + max(df["sharpness"]) * 0.02, f"{v:.4f}", ha="center", fontsize=8)
    else:
        with plt.rc_context(THESIS_RC):
            fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
            x = np.arange(len(df))
            width = 0.35
            ax_acc.bar(x - width / 2, df["test_acc"], width, label="test accuracy", color="#3498db")
            ax_acc_t = ax_acc.twinx()
            ax_acc_t.plot(x, df["epoka_stop"], "s--", color="#e74c3c", label="epoka zatrzymania")
            ax_acc.set_xticks(x)
            ax_acc.set_xticklabels([str(p) for p in df["patience"]])
            ax_acc.set_xlabel("patience $p$")
            ax_acc.set_ylabel("Dokładność testowa")
            ax_acc_t.set_ylabel("Epoka zatrzymania")
            ax_acc.set_title(f"ES — {ds}")
            ax_acc.grid(alpha=0.3, axis="y")
            bars_h, bars_l = ax_acc.get_legend_handles_labels()
            line_h, line_l = ax_acc_t.get_legend_handles_labels()
            ax_acc.legend(bars_h + line_h, bars_l + line_l, loc="upper left", fontsize=8)
            plt.tight_layout()
            _save_fig(fig_acc, acc_path)
            plt.close(fig_acc)
            panel_paths.append(acc_path)

            fig_sh, ax_sh = plt.subplots(figsize=(10, 5))
            ax_sh.bar(
                [str(p) for p in df["patience"]],
                df["sharpness"],
                color="#9b59b6",
                edgecolor="gray",
                label=r"$\hat{S}_P$",
            )
            ax_sh.set_xlabel("patience $p$")
            ax_sh.set_ylabel("Ostrość")
            ax_sh.set_title("Ostrość po ES")
            ax_sh.legend(loc="upper right", fontsize=8)
            ax_sh.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            _save_fig(fig_sh, sharp_path)
            plt.close(fig_sh)

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        x = np.arange(len(df))
        width = 0.35
        axes[0].bar(x - width / 2, df["test_acc"], width, color="#3498db")
        ax0b = axes[0].twinx()
        ax0b.plot(x, df["epoka_stop"], "s--", color="#e74c3c")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([str(p) for p in df["patience"]])
        axes[0].set_xlabel("patience $p$")
        axes[0].set_ylabel("Dokładność testowa")
        ax0b.set_ylabel("Epoka zatrzymania")
        axes[0].set_title(f"ES — {ds}")
        axes[0].grid(alpha=0.3, axis="y")
        axes[1].bar(
            [str(p) for p in df["patience"]], df["sharpness"],
            color="#9b59b6", edgecolor="gray",
        )
        axes[1].set_xlabel("patience $p$")
        axes[1].set_ylabel("Ostrość")
        axes[1].set_title("Ostrość po ES")
        axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle(f"Diagnostyka patience — {ds}", fontweight="bold")
    plt.tight_layout()

    if out_path:
        _save_fig(fig, out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _plot_val_acc_panel(
    histories: Dict[str, Dict],
    title_suffix: str = "",
    out_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ds = CFG["dataset"]
        for name, h in histories.items():
            if "val_acc" in h:
                ax.plot(
                    h["val_acc"], label=name, linewidth=1.8,
                    color=_method_color(name),
                )
        ax.set_title(f"Dokładność walidacyjna — {ds}{title_suffix}")
        ax.set_xlabel("Epoka")
        ax.set_ylabel("val accuracy")
        ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        if out_path:
            _save_fig(fig, out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig


def _plot_train_loss_panel(
    histories: Dict[str, Dict],
    out_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for name, h in histories.items():
            if "train_loss" in h:
                ax.plot(
                    h["train_loss"], label=name, linewidth=1.8,
                    color=_method_color(name),
                )
        ax.set_title("Strata treningowa (cross-entropy)")
        ax.set_xlabel("Epoka")
        ax.set_ylabel("train loss")
        ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        if out_path:
            _save_fig(fig, out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig


def _plot_sharpness_panel(
    histories: Dict[str, Dict],
    sharpness_by_method: Optional[Dict[str, float]] = None,
    out_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(histories.keys())
        sharps = []
        for n in names:
            if sharpness_by_method and n in sharpness_by_method:
                sharps.append(float(sharpness_by_method[n]))
            else:
                sharps.append(float(histories[n].get("sharpness", 0) or 0))
        ax.bar(
            range(len(names)), sharps,
            color=METHOD_COLORS[: len(names)], edgecolor="gray",
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(METHOD_SHORT[: len(names)], fontsize=10)
        ax.set_title(r"Ostrość $\hat{S}_P$ (końcowa)")
        ax.set_ylabel("sharpness")
        if sharps and max(sharps) > 0:
            ax.set_ylim(0, max(sharps) * 1.15)
            for i, v in enumerate(sharps):
                ax.text(i, v + max(sharps) * 0.02, f"{v:.4f}", ha="center", fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        plt.tight_layout()
        if out_path:
            _save_fig(fig, out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig


def plot_comparison(
    histories: Dict[str, Dict],
    out_path: Optional[str] = None,
    title_suffix: str = "",
    show: bool = True,
    sharpness_by_method: Optional[Dict[str, float]] = None,
    save_panels: bool = True,
) -> plt.Figure:
    """Wykres 3-panelowy + opcjonalnie trzy osobne pliki PNG (białe tło)."""
    panel_paths: Dict[str, Optional[str]] = {}
    if save_panels:
        panel_paths = {
            "val_acc": get_results_path("porownanie_val_acc.png"),
            "train_loss": get_results_path("porownanie_train_loss.png"),
            "sharpness": get_results_path("porownanie_sharpness.png"),
        }

    _plot_val_acc_panel(
        histories, title_suffix=title_suffix,
        out_path=panel_paths.get("val_acc"), show=False,
    )
    _plot_train_loss_panel(
        histories, out_path=panel_paths.get("train_loss"), show=False,
    )
    _plot_sharpness_panel(
        histories, sharpness_by_method=sharpness_by_method,
        out_path=panel_paths.get("sharpness"), show=False,
    )

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        ds = CFG["dataset"]

        for name, h in histories.items():
            if "val_acc" in h:
                axes[0].plot(
                    h["val_acc"], label=name,
                    color=_method_color(name),
                )
        axes[0].set_title(f"Dokładność walidacyjna — {ds}{title_suffix}")
        axes[0].set_ylabel("val accuracy")
        axes[0].set_xlabel("Epoka")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.25)

        for name, h in histories.items():
            if "train_loss" in h:
                axes[1].plot(
                    h["train_loss"], label=name,
                    color=_method_color(name),
                )
        axes[1].set_title("Strata treningowa (cross-entropy)")
        axes[1].set_ylabel("train loss")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.25)

        names = list(histories.keys())
        sharps = []
        for n in names:
            if sharpness_by_method and n in sharpness_by_method:
                sharps.append(float(sharpness_by_method[n]))
            else:
                sharps.append(float(histories[n].get("sharpness", 0) or 0))

        axes[2].bar(range(len(names)), sharps, color=METHOD_COLORS[: len(names)])
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(METHOD_SHORT[: len(names)], fontsize=9)
        axes[2].set_title(r"Ostrość $\hat{S}_P$ (końcowa)")
        axes[2].set_ylabel("sharpness")
        axes[2].grid(alpha=0.25, axis="y")
        if sharps and max(sharps) > 0:
            axes[2].set_ylim(0, max(sharps) * 1.15)
            for i, v in enumerate(sharps):
                axes[2].text(i, v + max(sharps) * 0.02, f"{v:.4f}", ha="center", fontsize=8)

        plt.tight_layout()
        if out_path:
            _save_fig(fig, out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig


def plot_final_summary(
    multi_seed: Dict[str, List],
    out_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Wykres słupkowy mean ± std — multi-seed (do pracy dyplomowej)."""
    methods = [m for m, runs in multi_seed.items() if runs]
    acc_means, acc_stds, sharp_means, sharp_stds = [], [], [], []

    for m in methods:
        acc_means.append(np.mean([r["test_acc"] for r in multi_seed[m]]))
        acc_stds.append(np.std([r["test_acc"] for r in multi_seed[m]]))
        sharp_means.append(np.mean([r["sharpness_mean"] for r in multi_seed[m]]))
        sharp_stds.append(np.std([r["sharpness_mean"] for r in multi_seed[m]]))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    x = np.arange(len(methods))
    labels = ["ES", "SGDR", "SWA", "Noise\nInj.", "MU\nEscape"][: len(methods)]
    ds = CFG["dataset"]
    n_seeds = len(multi_seed[methods[0]]) if methods else 0

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(x, acc_means, yerr=acc_stds, capsize=5, color=colors[: len(methods)],
                edgecolor="gray", alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_title(f"Dokładność testowa — {ds}")
    axes[0].set_ylabel("Accuracy")
    for i, (m, s) in enumerate(zip(acc_means, acc_stds)):
        axes[0].text(i, m + s + 0.005, f"{m:.4f}", ha="center", fontsize=8)

    axes[1].bar(x, sharp_means, yerr=sharp_stds, capsize=5, color=colors[: len(methods)],
                edgecolor="gray", alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_title(f"Ostrość — {ds}")
    axes[1].set_ylabel("Perturbation sharpness")
    for i, (m, s) in enumerate(zip(sharp_means, sharp_stds)):
        axes[1].text(i, m + s + 0.0003, f"{m:.4f}", ha="center", fontsize=8)

    fig.suptitle(f"Porównanie metod ({n_seeds} ziarna, mean ± std)", fontweight="bold")
    plt.tight_layout()

    if out_path:
        _save_fig(fig, out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def load_results_from_checkpoints(seeds: List[int]) -> Dict[str, List]:
    """Odtwarza słownik wyników z checkpointów (bez ponownego treningu)."""
    from .experiments import METHOD_PIPELINE, _result_from_checkpoint
    from .data import get_loaders
    from .config import get_device

    device = get_device()
    all_results: Dict[str, List] = {name: [] for name, _, _ in METHOD_PIPELINE}

    for seed in seeds:
        train_loader, val_loader, test_loader, _, num_classes = get_loaders(seed=seed)
        for name, prefix, _ in METHOD_PIPELINE:
            row = _result_from_checkpoint(
                name, seed, prefix, device, val_loader, test_loader,
                quiet=True,
            )
            if row:
                row.pop("_model", None)
                all_results[name].append(row)
    return all_results


def regenerate_plots_from_checkpoints(
    seeds: Optional[List[int]] = None,
    show: bool = True,
) -> Dict[str, str]:
    """
    Przelicza wykresy z checkpointów (białe tło + panele osobno).
    Nie uruchamia treningu — tylko kilka minut na ostrość/ewaluację.
    """
    seeds = seeds or CFG["seeds"]
    all_results = load_results_from_checkpoints(seeds)
    missing = [m for m, runs in all_results.items() if len(runs) < len(seeds)]
    if missing:
        print("Uwaga: brak checkpointów dla:", ", ".join(missing))
    return save_and_show_results(all_results, seeds[-1], show=show)


def save_and_show_results(
    all_results: Dict[str, List],
    last_seed: int,
    show: bool = True,
) -> Dict[str, str]:
    """
    Zapisuje CSV/JSON/wykresy i opcjonalnie pokazuje w notatniku.
    Zwraca słownik ścieżek do plików.
    """
    import json

    from .storage import get_results_path

    from .trajectory_analysis import export_trajectory_summary

    paths: Dict[str, str] = {}
    df = build_summary_table(all_results)

    csv_path = get_results_path("tabela_finalna.csv")
    df.to_csv(csv_path, index=False)
    save_to_drive(csv_path, f"results/{CFG['dataset']}/tabela_finalna.csv")
    paths["csv"] = csv_path

    export = {
        m: [{k: v for k, v in r.items() if k != "history"} for r in runs]
        for m, runs in all_results.items()
    }
    json_path = get_results_path("surowe_wyniki.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    save_to_drive(json_path, f"results/{CFG['dataset']}/surowe_wyniki.json")
    paths["json"] = json_path

    histories = {
        m: all_results[m][-1]["history"]
        for m in all_results
        if all_results[m]
    }
    sharpness_map = {
        m: all_results[m][-1].get("sharpness_mean", 0.0)
        for m in all_results
        if all_results[m]
    }
    cmp_path = get_results_path("porownanie_wykres.png")
    plot_comparison(
        histories, cmp_path,
        title_suffix=f" (seed {last_seed})",
        show=show,
        sharpness_by_method=sharpness_map,
        save_panels=True,
    )
    paths["porownanie"] = cmp_path
    paths["porownanie_val_acc"] = get_results_path("porownanie_val_acc.png")
    paths["porownanie_train_loss"] = get_results_path("porownanie_train_loss.png")
    paths["porownanie_sharpness"] = get_results_path("porownanie_sharpness.png")

    traj_path = export_trajectory_summary(last_seed)
    paths["trajektorie"] = traj_path

    if any(len(runs) > 1 for runs in all_results.values()):
        final_path = get_results_path("wykres_finalny.png")
        plot_final_summary(all_results, final_path, show=show)
        paths["finalny"] = final_path

    if show:
        print("\n" + "=" * 72)
        print(df.to_string(index=False))
        print("=" * 72)
        print(f"\nZapisano w: {os.path.dirname(csv_path)}")

    return paths
