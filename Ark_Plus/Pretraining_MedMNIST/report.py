"""
Results report for an Ark+ MedMNIST run: summary table, AUC comparison chart,
validation-loss curves. Used by the notebook; also runs from the command line:

    python report.py --run_dir <.../Models/swin_base_<exp>/Ark_Plus_.../<exp>> --out_dir report
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Palette (validated: blue/orange pass CVD and normal-vision separation).
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9a988f"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"
ARK = "#2a78d6"        # Ark+ (joint)            slot 1
BASE_P2 = "#eb6834"    # your individual runs    slot 2
BASE_P1 = MUTED        # published individual    reference, gray
BASE_IND = "#1baf7a"   # individual, same Ark+ pipeline   slot 3
LABELS = {"P1": "P1 published (individual)", "P2": "P2 your runs (individual)",
          "IND": "Same pipeline, individual"}

DATASET_ORDER = ["ChestMNIST", "DermaMNIST", "RetinaMNIST", "BreastMNIST"]


def _style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    })


def load_run(run_dir):
    final = json.load(open(os.path.join(run_dir, "final_results.json")))
    hist_p = os.path.join(run_dir, "history.json")
    history = json.load(open(hist_p)) if os.path.isfile(hist_p) else []
    return final, history


def summary_table(final, baselines=None):
    """baselines: {"P1": {"ChestMNIST": 0.768, ...}, "P2": {...}}  (None values are skipped)."""
    baselines = baselines or {}
    rows = []
    for d in [x for x in DATASET_ORDER if x in final["datasets"]] + [x for x in final["datasets"] if x not in DATASET_ORDER]:
        t, s = final["teacher"][d], final["student"][d]
        row = {"Dataset": d, "Task": t["task_type"].replace(" classification", ""), "Test images": t["n_test"]}
        for name, vals in baselines.items():
            v = (vals or {}).get(d)
            row["{} AUC".format(name)] = v
        row["Ark+ teacher AUC"] = t["auc"]
        row["Ark+ student AUC"] = s["auc"]
        row["Ark+ teacher ACC"] = t["acc"]
        for name, vals in baselines.items():
            v = (vals or {}).get(d)
            row["Δ vs {}".format(name)] = (t["auc"] - v) if v is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def styled(df):
    fmt = {c: "{:.4f}" for c in df.columns if "AUC" in c or "ACC" in c}
    fmt.update({c: "{:+.4f}" for c in df.columns if c.startswith("Δ")})

    def color_delta(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return "color: #1a7f37; font-weight: 600" if v >= 0 else "color: #c62828; font-weight: 600"

    sty = df.style.format(fmt, na_rep="n/a").hide(axis="index")
    delta_cols = [c for c in df.columns if c.startswith("Δ")]
    if delta_cols:
        sty = sty.map(color_delta, subset=delta_cols)
    return sty.set_table_styles([
        {"selector": "th", "props": "background:#f0efec; color:#0b0b0b; font-weight:600; padding:6px 10px; text-align:left"},
        {"selector": "td", "props": "padding:6px 10px"},
    ])


def plot_auc(final, baselines=None, out=None):
    """Dot plot: one row per dataset, baseline dot(s) and Ark+ dot, joined by a line.
    Dots, not bars, so the axis can zoom to the AUC range without a truncated-bar lie."""
    _style()
    baselines = {k: v for k, v in (baselines or {}).items() if v and any(x is not None for x in v.values())}
    ds = [x for x in DATASET_ORDER if x in final["datasets"]]
    ark = [final["teacher"][d]["auc"] for d in ds]
    fig, ax = plt.subplots(figsize=(8.2, 0.9 + 0.75 * len(ds)))
    y = np.arange(len(ds))[::-1]
    colors = {"P1": BASE_P1, "P2": BASE_P2, "IND": BASE_IND}
    vals_all = list(ark)
    for name, vals in baselines.items():
        xs = [vals.get(d) for d in ds]
        for yi, xb, xa in zip(y, xs, ark):
            if xb is not None:
                ax.plot([xb, xa], [yi, yi], color=GRID, lw=2, zorder=1)
        pts = [(yi, xb) for yi, xb in zip(y, xs) if xb is not None]
        if pts:
            ax.scatter([p[1] for p in pts], [p[0] for p in pts], s=80, color=colors.get(name, MUTED),
                       edgecolor=SURFACE, linewidth=2, zorder=2, label=LABELS.get(name, name))
            vals_all += [p[1] for p in pts]
    ax.scatter(ark, y, s=95, color=ARK, edgecolor=SURFACE, linewidth=2, zorder=3, label="Ark+ joint (teacher)")
    ref = next((r for r in ["IND", "P2", "P1"] if r in baselines), None)
    for yi, d, xa in zip(y, ds, ark):
        txt = "{:.3f}".format(xa)
        if ref and baselines[ref].get(d) is not None:
            delta = xa - baselines[ref][d]
            txt += "  ({:+.3f} vs {})".format(delta, ref)
        ax.annotate(txt, (xa, yi), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(ds)
    lo, hi = min(vals_all), max(vals_all)
    pad = max(0.02, (hi - lo) * 0.15)
    ax.set_xlim(max(0, lo - pad), min(1.0, hi + pad * 3.5))
    ax.set_ylim(-0.6, len(ds) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test AUC (mean over classes)")
    ax.set_title("Ark+ on 4 MedMNIST datasets: test AUC", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2 if len(ds) > 2 else -0.35), ncol=2, fontsize=9.5)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=160, bbox_inches="tight")
    return fig


def plot_history(history, final=None, out=None):
    """Small multiples: validation loss per dataset (each on its own scale),
    best epoch marked. A fifth strip shows the average used for early stopping."""
    _style()
    if not history:
        return None
    ds = [x for x in DATASET_ORDER if x in history[0]["val_loss"]]
    ep = [h["epoch"] for h in history]
    best = final["best_epoch"] if final else int(np.argmin([h["val_metric"] for h in history]))
    fig, axes = plt.subplots(1, len(ds) + 1, figsize=(3.0 * (len(ds) + 1), 2.8), sharex=True)
    series = [(d, [h["val_loss"][d] for h in history]) for d in ds] + [("Average (early-stop metric)", [h["val_metric"] for h in history])]
    for ax, (name, v) in zip(axes, series):
        color = ARK if not name.startswith("Average") else INK_2
        ax.plot(ep, v, color=color, lw=2, marker="o" if len(ep) <= 15 else None, ms=4)
        if best in ep:
            bi = ep.index(best)
            ax.axvline(best, color=MUTED, lw=1, ls="--")
            ax.scatter([best], [v[bi]], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate("best {:.4f}".format(v[bi]), (best, v[bi]), xytext=(8, 10), textcoords="offset points",
                        ha="left", fontsize=8.5, color=INK,
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
        ax.set_title(name, loc="left", fontsize=10.5)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("validation loss")
    stop = " | stopped early" if final and final.get("stopped_early") else ""
    fig.suptitle("Validation loss per epoch (best epoch {}{})".format(best, stop), x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=160, bbox_inches="tight")
    return fig


def write_report(run_dir, out_dir, baselines=None):
    os.makedirs(out_dir, exist_ok=True)
    final, history = load_run(run_dir)
    df = summary_table(final, baselines)
    df.to_csv(os.path.join(out_dir, "results_table.csv"), index=False)
    plot_auc(final, baselines, os.path.join(out_dir, "auc_comparison.png"))
    plot_history(history, final, os.path.join(out_dir, "val_loss_curves.png"))
    return final, history, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default="report")
    ap.add_argument("--baselines_json", default=None, help='{"P1": {"ChestMNIST": 0.77, ...}, "P2": {...}}')
    a = ap.parse_args()
    b = json.load(open(a.baselines_json)) if a.baselines_json else None
    _, _, df = write_report(a.run_dir, a.out_dir, b)
    print(df.to_string(index=False))
