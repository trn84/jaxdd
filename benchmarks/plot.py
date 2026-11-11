"""The README's speed figure and table, from `benchmarks/results/*.json` (`run.py`).

    python benchmarks/plot.py                       # reference: the SimSS run on the most cores
    python benchmarks/plot.py --reference "my server"

The entries, each the current measurement: SimSS 5.36 on all cores of the reference machine,
jaxdd on all cores of the same machine (one process per core), and jaxdd on every GPU measured
(its best batch), each with its factor over SimSS. A GPU's FP64 rate against its FP32 rate is
named beside it: jaxdd computes in FP64, which data-centre cards run at half their FP32 rate
and consumer cards at 1/64.
Writes `benchmarks/speedup-light.svg` and `-dark.svg` and the block between
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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
README = ROOT / "README.md"
BEGIN, END = "<!-- benchmark:begin -->", "<!-- benchmark:end -->"

#: The validated reference palette (categorical slots 1–3, light and dark steps), its ink and
#: chrome. Every point is labelled, and the table holds every number.
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
NAMES = {"simss": "SimSS 5.36", "cpu": "jaxdd", "gpu": "jaxdd"}
#: FP64 against FP32 throughput, by a word in the GPU's name (the vendors' data sheets).
FP64_RATIO = {"A100": "1/2", "H100": "1/2", "H200": "1/2", "V100": "1/2",
              "GeForce": "1/64", "RTX": "1/64"}  # fmt: skip


def load() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]


def best(report: dict, kind: str) -> dict | None:
    """A machine's best result of `kind`: "simss", or jaxdd on the "cpu" or the "gpu"."""
    rows = [r for r in report["results"]
            if (r["solver"] == "simss" if kind == "simss" else r["device"] == kind)]  # fmt: skip
    return max(rows, key=lambda r: r["runs_per_s"]) if rows else None


def entries_of(reports: list[dict], label: str | None) -> list[tuple[str, dict, dict]]:
    """SimSS on the reference machine (the named one, else the one on the most cores), jaxdd on
    that machine's cores, jaxdd on every GPU, slowest first."""
    runs = [(r, best(r, "simss")) for r in reports if best(r, "simss")]
    if label:
        runs = [(r, b) for r, b in runs if r["label"] == label]
    if not runs:
        raise SystemExit(f"no SimSS run in {RESULTS}" + (f" labelled {label!r}" if label else ""))
    ref, simss = max(runs, key=lambda rb: (rb[1]["procs"], rb[1]["runs_per_s"]))
    out = [("simss", ref, simss)]
    if best(ref, "cpu"):
        out.append(("cpu", ref, best(ref, "cpu")))
    gpus = [("gpu", r, best(r, "gpu")) for r in reports if best(r, "gpu")]
    return out + sorted(gpus, key=lambda e: e[2]["runs_per_s"])


def describe(kind: str, report: dict, row: dict) -> tuple[str, str]:
    """`(solver, hardware and setting)` of an entry."""
    if kind != "gpu":
        return NAMES[kind], f"{report['machine']['cpu']}, {row['procs']} processes"
    name = row["gpu"].removeprefix("NVIDIA ")
    ratio = next((v for k, v in FP64_RATIO.items() if k in name), None)
    fp64 = f", FP64 at {ratio} of FP32" if ratio else ""
    return NAMES[kind], f"{name}, batch {row['batch']}{fp64}"


def _rate(value: float) -> str:
    return f"{value:.3g}" if value < 100 else f"{value:,.0f}"


def figure(entries: list[tuple[str, dict, dict]], theme: str, path: Path) -> None:
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 10, "svg.fonttype": "none"})  # fmt: skip
    fig, ax = plt.subplots(figsize=(8.0, 0.7 + 0.75 * len(entries)), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ref_rate = entries[0][2]["runs_per_s"]
    rates = [row["runs_per_s"] for _, _, row in entries]
    for y, (kind, _, row) in enumerate(entries):
        rate = row["runs_per_s"]
        ax.plot(rate, y, MARKERS[kind], color=t["series"][kind], markersize=10,
                markeredgecolor=t["surface"], markeredgewidth=2, zorder=3)  # fmt: skip
        label = _rate(rate) + (f"  ×{rate / ref_rate:,.0f}" if y else "")
        ax.annotate(label, (rate, y), xytext=(0, 9), textcoords="offset points", ha="center",
                    va="bottom", color=t["ink2"], fontsize=9.5)  # fmt: skip
    ax.set_xscale("log")
    # Whole decades, at least three, labelled as plain numbers.
    low = 10 ** np.floor(np.log10(min(rates) / 2))
    high = max(10 ** np.ceil(np.log10(max(rates) * 2)), low * 1e3)
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.6, len(entries) - 0.25)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(["\n".join(describe(k, r, row)) for k, r, row in entries], color=t["ink"])
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
    fig.tight_layout()
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def table(entries: list[tuple[str, dict, dict]]) -> str:
    ref_rate = entries[0][2]["runs_per_s"]
    lines = ["| Solver | Hardware | Curves per second | × SimSS |", "|---|---|---|---|"]
    for kind, report, row in entries:
        solver, how = describe(kind, report, row)
        lines.append(f"| {solver} | {how} | {_rate(row['runs_per_s'])} "
                     f"| {row['runs_per_s'] / ref_rate:,.0f} |")  # fmt: skip
    return "\n".join(lines)


def readme_block(entries: list[tuple[str, dict, dict]]) -> str:
    picture = ('<picture>\n  <source media="(prefers-color-scheme: dark)" '
               'srcset="benchmarks/speedup-dark.svg">\n  <img src="benchmarks/speedup-light.svg" '
               'alt="J-V curves per second: SimSS and jaxdd on CPU cores, jaxdd on GPUs">\n'
               "</picture>")  # fmt: skip
    ref = entries[0][1]
    acc = ref.get("accuracy")
    note = (
        "The same devices for every solver (`benchmarks/run.py`: the three committed SimSS "
        "cases, traps, ions and doping drawn at random), each measured with the current code. "
        "jaxdd computes in FP64: a data-centre GPU runs it at half its FP32 rate, a consumer "
        "card at 1/64, which is why a gaming GPU is barely faster than a desktop CPU here."
    )
    if acc:
        note += (f" jaxdd's curves agree with SimSS's to {acc['max']:.1e} of the current scale "
                 f"on {acc['devices']} of the devices.")  # fmt: skip
    note += " How to get the most out of each: [`docs/hardware.md`](docs/hardware.md)."
    return f"{BEGIN}\n{picture}\n\n{note}\n\n{table(entries)}\n{END}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", default=None, help="the label of the SimSS machine")
    args = parser.parse_args(argv)
    entries = entries_of(load(), args.reference)
    for theme in THEMES:
        figure(entries, theme, ROOT / "benchmarks" / f"speedup-{theme}.svg")
    text = README.read_text(encoding="utf-8")
    block = readme_block(entries)
    if BEGIN in text:
        text = re.sub(f"{re.escape(BEGIN)}.*?{re.escape(END)}", lambda _: block, text, flags=re.S)
    else:
        text = text.replace("## Licence", f"## Speed\n\n{block}\n\n## Licence", 1)
    README.write_bytes(text.encode("utf-8"))
    print(f"{len(entries)} entries; figure and README table written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
