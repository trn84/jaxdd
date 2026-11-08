"""The README's speed figure and table, from every `benchmarks/results/*.json` (`run.py`).

    python benchmarks/plot.py                       # reference: the SimSS run on the most cores
    python benchmarks/plot.py --reference "my server"

One row per machine, J-V curves per second on a log axis: SimSS on all its cores, jaxdd on all
of them (one process per core), jaxdd on the GPU (its best batch). Each jaxdd point carries its
factor over the reference SimSS run — the fastest way to get the same curves on hardware you
already have. Writes `benchmarks/speedup-light.svg` and `-dark.svg` and the block between
`<!-- benchmark:begin -->` and `<!-- benchmark:end -->` in README.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
README = ROOT / "README.md"
BEGIN, END = "<!-- benchmark:begin -->", "<!-- benchmark:end -->"

#: The validated reference palette (categorical slots 1–3, light and dark steps), its ink and
#: chrome. Aqua sits below 3:1 on the light surface: every point is labelled, and the table
#: holds every number.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": {"simss": "#2a78d6", "cpu": "#eb6834", "gpu": "#1baf7a"}},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835",
             "series": {"simss": "#3987e5", "cpu": "#d95926", "gpu": "#199e70"}},
}  # fmt: skip
#: Marker shape per series as well as hue, so identity never rests on colour alone.
MARKERS = {"simss": "o", "cpu": "s", "gpu": "D"}
NAMES = {"simss": "SimSS 5.36, all cores", "cpu": "jaxdd, all CPU cores", "gpu": "jaxdd, GPU"}
#: Where a point's label goes: the CPU's below its point, the others above, so that a CPU and a
#: GPU point close together on the axis keep their labels apart.
LABEL_BELOW = {"cpu"}


def load() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]


def points(report: dict) -> dict[str, dict]:
    """A machine's best result per series."""
    best: dict[str, dict] = {}
    for row in report["results"]:
        key = "simss" if row["solver"] == "simss" else row["device"]
        if key not in best or row["runs_per_s"] > best[key]["runs_per_s"]:
            best[key] = row
    return best


def hardware(report: dict) -> str:
    machine = report["machine"]
    parts = [machine["gpu"]] if machine.get("gpu") else []
    return " · ".join([*parts, f"{machine['logical_cores']} logical cores"])


def reference(reports: list[dict], label: str | None) -> tuple[dict, dict] | None:
    """The SimSS run every factor is over: the named machine's, or the one on the most cores."""
    runs = [(r, points(r)["simss"]) for r in reports if "simss" in points(r)]
    if label:
        runs = [(r, s) for r, s in runs if r["label"] == label]
    if not runs:
        return None
    return max(runs, key=lambda rs: (rs[1]["procs"], rs[1]["runs_per_s"]))


def _rate(value: float) -> str:
    return f"{value:.3g}" if value < 100 else f"{value:,.0f}"


def figure(reports: list[dict], ref: tuple[dict, dict] | None, theme: str, path: Path) -> None:
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 10, "svg.fonttype": "none"})  # fmt: skip
    rows = sorted(reports, key=lambda r: max(p["runs_per_s"] for p in points(r).values()))
    fig, ax = plt.subplots(figsize=(8.0, 1.1 + 0.85 * len(rows)), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ref_rate = ref[1]["runs_per_s"] if ref else None
    lows, highs = [], []
    for y, report in enumerate(rows):
        for key, row in points(report).items():
            rate = row["runs_per_s"]
            lows.append(rate)
            highs.append(rate)
            ax.plot(rate, y, MARKERS[key], color=t["series"][key], markersize=9,
                    markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)  # fmt: skip
            label = _rate(rate)
            if ref_rate and key != "simss":
                label += f"  ×{rate / ref_rate:,.0f}"
            below = key in LABEL_BELOW
            ax.annotate(label, (rate, y), xytext=(0, -10 if below else 9),
                        textcoords="offset points", ha="center", va="top" if below else "bottom",
                        color=t["ink2"], fontsize=9)  # fmt: skip
    if ref_rate:
        ax.axvline(ref_rate, color=t["muted"], linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.annotate(
            "reference",
            (ref_rate, len(rows) - 0.35),
            xytext=(4, 0),
            textcoords="offset points",
            color=t["muted"],
            fontsize=8.5,
            va="center",
        )
    ax.set_xscale("log")
    # Whole decades, at least three, labelled as plain numbers: rates span 0.1 to 10⁴ per second.
    low = 10 ** np.floor(np.log10(min(lows) / 2))
    high = max(10 ** np.ceil(np.log10(max(highs) * 2)), low * 1e3)
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['label']}\n{hardware(r)}" for r in rows], color=t["ink"])
    for tick in ax.get_yticklabels():
        tick.set_linespacing(1.3)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=t["muted"], labelcolor=t["ink2"])
    ax.xaxis.grid(True, which="major", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.set_xlabel("J-V curves per second (61 voltages each), log scale", color=t["ink2"])
    present = {key for report in rows for key in points(report)}
    handles = [Line2D([], [], marker=MARKERS[k], linestyle="", color=t["series"][k],
                      markersize=8, label=NAMES[k]) for k in MARKERS if k in present]  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    legend = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9,
                        handletextpad=0.3, columnspacing=1.6)  # fmt: skip
    for text in legend.get_texts():
        text.set_color(t["ink"])
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def table(reports: list[dict], ref: tuple[dict, dict] | None) -> str:
    ref_rate = ref[1]["runs_per_s"] if ref else None
    lines = ["| Machine | Solver | Processes / batch | Curves per second | × reference SimSS "
             "| jaxdd against SimSS (max ΔJ / scale) |", "|---|---|---|---|---|---|"]  # fmt: skip
    for report in sorted(reports, key=lambda r: r["label"]):
        acc = report.get("accuracy")
        acc_text = f"{acc['max']:.1e} on {acc['devices']} devices" if acc else "—"
        for key, row in sorted(points(report).items(), key=lambda kv: list(MARKERS).index(kv[0])):
            how = f"batch {row['batch']}" if key == "gpu" else f"{row['procs']} processes"
            factor = f"{row['runs_per_s'] / ref_rate:,.1f}" if ref_rate else "—"
            lines.append(f"| {report['label']} ({hardware(report)}) | {NAMES[key]} | {how} "
                         f"| {_rate(row['runs_per_s'])} | {factor} "
                         f"| {acc_text if key == 'simss' else ''} |")  # fmt: skip
    return "\n".join(lines)


def readme_block(reports: list[dict], ref: tuple[dict, dict] | None) -> str:
    picture = ('<picture>\n  <source media="(prefers-color-scheme: dark)" '
               'srcset="benchmarks/speedup-dark.svg">\n  <img src="benchmarks/speedup-light.svg" '
               'alt="J-V curves per second: SimSS and jaxdd on CPU and GPU, per machine">\n'
               "</picture>")  # fmt: skip
    note = (
        "The same devices for every solver (`benchmarks/run.py`: the three committed SimSS "
        "cases, traps, ions and doping drawn at random). Run it on a machine of yours and "
        "send the JSON in `benchmarks/results/`; `benchmarks/plot.py` redraws this."
    )
    if not ref:
        note += " No machine with SimSS measured yet, so no factor over it."
    if ref:
        note += (f" Factors over SimSS on {ref[0]['label']} ({ref[1]['procs']} processes, "
                 f"{_rate(ref[1]['runs_per_s'])} curves/s).")  # fmt: skip
    return f"{BEGIN}\n{picture}\n\n{note}\n\n{table(reports, ref)}\n{END}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reference", default=None, help="the label of the SimSS run to compare to"
    )
    args = parser.parse_args(argv)
    reports = load()
    if not reports:
        raise SystemExit(f"no results in {RESULTS}: run benchmarks/run.py first")
    ref = reference(reports, args.reference)
    for theme in THEMES:
        figure(reports, ref, theme, ROOT / "benchmarks" / f"speedup-{theme}.svg")
    text = README.read_text(encoding="utf-8")
    block = readme_block(reports, ref)
    if BEGIN in text:
        text = re.sub(f"{re.escape(BEGIN)}.*?{re.escape(END)}", lambda _: block, text, flags=re.S)
    else:
        text = text.replace("## Licence", f"## Speed\n\n{block}\n\n## Licence", 1)
    README.write_bytes(text.encode("utf-8"))
    print(f"{len(reports)} machines; figure and README table written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
