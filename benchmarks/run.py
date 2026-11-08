"""jaxdd against SimSS on this machine: J-V curves per second, for `benchmarks/plot.py`.

    python benchmarks/run.py --label "A100 pod"                                  # jaxdd: GPU, CPU
    python benchmarks/run.py --label "EPYC 7502 server" --simss-docker fauxpv/simss:5.36
    python benchmarks/run.py --label "laptop" --simss simss --simss-fixed simss_fixed_ions

**The workload** is `validation/bench.py`'s: the three committed SimSS cases (no ions; 1e17 cm⁻³
frozen after a pre-bias; the same free), with the trap density, the ion density and the
transport layers' doping drawn at random from a fixed seed, at 61 voltages (−0.2 to 1.3 V).
One *run* is one J-V curve. Every solver gets the same devices, as SIMsalabim input files for
SimSS and built from those same inputs for jaxdd.

**What is measured**, each part only where the machine has it:

* `gpu` — jaxdd on JAX's GPU, one process, at each `--gpu-batches` size; compilation and a
  first batch excluded, then `--repeats` batches timed. The batch is `--gpu-unique` distinct
  devices tiled to its size (building a device is host time, ~25 ms, not the solver's).
* `cpu` — jaxdd on the CPU the way it runs fastest there: one single-threaded process per core
  (`--cpu-procs`, default every logical core), each solving `--cpu-batch` devices
  `--repeats` times after its own compilation, all started together; runs over wall time.
* `simss` — SimSS 5.36, one process per run, `--simss-procs` at once (default every logical
  core): native binaries (`--simss`, and `--simss-fixed` for the frozen-ion case, FauxPV's
  patched `simss_fixed_ions`), or a Docker image that has both (`--simss-docker`; its start,
  about a second, is in the time). Runs over wall time.
* `accuracy` — where SimSS ran: jaxdd (with SimSS's Bernoulli series, DIFFERENCES.md §2) on
  the same devices against SimSS's curves, `max |ΔJ|` over the current scale per device.
  SimSS's early stops (DIFFERENCES.md §1) show up here as SimSS's error, not jaxdd's.

Writes `benchmarks/results/<label>.json`; `benchmarks/plot.py` turns all of them into the
README's figure and table.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
RESULTS = ROOT / "benchmarks" / "results"
V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.025), 6)
SEED = 20261107
#: XLA's own threads do not help this solver on a CPU (tiny 3×3 blocks in a sequential sweep):
#: one single-threaded process per core is its fastest use of a CPU (NEXT.md).
SINGLE_THREAD_XLA = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
#: Devices compared in `accuracy`, at most.
ACCURACY_DEVICES = 48
#: A curve differs where `max |ΔJ|` exceeds this share of its current scale (jaxdd's gate).
ACCURACY_TOL = 1e-3


# ------------------------------------------------------------------------------- the machine
def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _memory_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _jaxdd_commit() -> str | None:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                              capture_output=True, text=True).stdout.strip()  # fmt: skip

    try:
        commit = git("rev-parse", "--short=12", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + (" with local changes" if dirty else "")


# ------------------------------------------------------------------------------- jaxdd
def _lockstep(its: np.ndarray) -> dict:
    """Newton steps per voltage: the mean the runs need against the per-voltage maximum the
    batch pays under `vmap` (PLAN.md §3.4)."""
    needed, paid = its.mean(axis=0).sum(), its.max(axis=0).sum()
    return {"steps_needed": float(needed / its.shape[1]), "steps_paid": float(paid / its.shape[1]),
            "lockstep": float(paid / needed)}  # fmt: skip


def measure_gpu(batches: list[int], unique: int, repeats: int) -> list[dict]:
    import jax
    from bench import variants

    from jaxdd.batch import prepare, solve_batch

    base = variants(min(unique, max(batches)), SEED)
    out = []
    for size in batches:
        devices = [base[k % len(base)] for k in range(size)]
        batch = prepare(devices)
        start = time.perf_counter()
        J, ok, its = solve_batch(batch, V)
        J.block_until_ready()
        first = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(repeats):
            J, ok, its = solve_batch(batch, V)
            J.block_until_ready()
        seconds = time.perf_counter() - start
        runs = size * repeats
        row = {"solver": "jaxdd", "device": "gpu", "gpu": jax.devices()[0].device_kind,
               "batch": size, "runs": runs, "seconds": seconds, "runs_per_s": runs / seconds,
               "first_batch_s": first, "converged": float(np.asarray(ok).mean()),
               **_lockstep(np.asarray(its))}  # fmt: skip
        print(f"gpu batch {size}: {row['runs_per_s']:.1f} runs/s (lockstep x{row['lockstep']:.2f})")
        out.append(row)
    return out


def _cpu_worker(index: int, batch: int, repeats: int, barrier, queue) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = SINGLE_THREAD_XLA
    from bench import variant_setups

    from jaxdd.batch import prepare, solve_batch
    from jaxdd.device import build_device

    setups = variant_setups(batch * (index + 1), SEED)[batch * index :]
    devices = prepare([build_device(s) for s in setups])
    J, ok, its = solve_batch(devices, V)  # compiles
    J.block_until_ready()
    barrier.wait()
    start = time.perf_counter()
    for _ in range(repeats):
        J, ok, its = solve_batch(devices, V)
        J.block_until_ready()
    queue.put((time.perf_counter() - start, float(np.asarray(ok).mean()), np.asarray(its)))


def measure_cpu(procs: int, batch: int, repeats: int) -> dict:
    context = multiprocessing.get_context("spawn")
    barrier, queue = context.Barrier(procs + 1), context.Queue()
    workers = [context.Process(target=_cpu_worker, args=(i, batch, repeats, barrier, queue))
               for i in range(procs)]  # fmt: skip
    for w in workers:
        w.start()
    barrier.wait()  # every worker compiled
    start = time.perf_counter()
    results = [queue.get() for _ in workers]
    wall = time.perf_counter() - start
    for w in workers:
        w.join()
    runs = procs * batch * repeats
    its = np.concatenate([r[2] for r in results])
    row = {"solver": "jaxdd", "device": "cpu", "procs": procs, "batch": batch, "runs": runs,
           "seconds": wall, "runs_per_s": runs / wall,
           "per_process_runs_per_s": float(np.mean([batch * repeats / r[0] for r in results])),
           "converged": float(np.mean([r[1] for r in results])), **_lockstep(its)}  # fmt: skip
    print(f"cpu {procs} processes: {row['runs_per_s']:.1f} runs/s")
    return row


# ------------------------------------------------------------------------------- SimSS
def write_simss_inputs(setup, target: Path) -> None:
    """A variant as SimSS's files: its case's files, each value the variant changed rewritten
    in place (SimSS reads its parameter files by position, so the files keep their layout)."""
    from jaxdd.simss_input import read_parameter_file

    shutil.copytree(setup.directory, target)
    (target / "JV.dat").unlink(missing_ok=True)
    files = [("simulation_setup.txt", setup.raw)]
    files += [(setup.raw[f"l{k}"], layer.raw) for k, layer in enumerate(setup.layers, start=1)]
    for name, values in files:
        path = target / name
        text, before = path.read_text(), read_parameter_file(path)
        for key, value in values.items():
            if before.get(key) == value:
                continue
            text, count = re.subn(rf"^({re.escape(key)}\s*=\s*)\S+", rf"\g<1>{value}", text,
                                  flags=re.M)  # fmt: skip
            if count != 1:
                raise KeyError(f"{path}: {key}")
        path.write_text(text, newline="\n")


def _binary(setup, plain: str, fixed: str) -> str:
    """The frozen-ion case needs FauxPV's patched SimSS (`fixIons` held in Poisson too)."""
    return fixed if setup.i("preCond") and setup.i("fixIons") else plain


def measure_simss(n: int, procs: int, plain: str, fixed: str, docker: str | None,
                  docker_cmd: list[str]) -> tuple[dict, dict[int, np.ndarray]]:  # fmt: skip
    from bench import variant_setups

    setups = variant_setups(n, SEED)
    work = Path(tempfile.mkdtemp(prefix="jaxdd_bench_"))
    try:
        for k, setup in enumerate(setups):
            write_simss_inputs(setup, work / f"r{k:05d}")
            (work / f"r{k:05d}" / "bin").write_text(_binary(setup, plain, fixed))
        start = time.perf_counter()
        if docker:
            mount = str(work)
            if docker_cmd[0] == "wsl":  # Docker inside WSL: the path as WSL sees it
                mount = subprocess.run(["wsl", "-e", "wslpath", "-a", mount], check=True,
                                       capture_output=True, text=True).stdout.strip()  # fmt: skip
            one_run = "cd {} && $(cat bin) simulation_setup.txt > log.out 2>&1; true"
            script = f"ls -d /work/r* | xargs -P {procs} -I{{}} sh -c '{one_run}'"
            user = ["-u", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
            subprocess.run([*docker_cmd, "run", "--rm", *user, "-v", f"{mount}:/work",
                            "--entrypoint", "sh", docker, "-c", script], check=True)  # fmt: skip
        else:

            def one(k: int) -> None:
                directory = work / f"r{k:05d}"
                with open(directory / "log.out", "w") as log:
                    subprocess.run([(directory / "bin").read_text(), "simulation_setup.txt"],
                                   cwd=directory, stdout=log, stderr=subprocess.STDOUT)  # fmt: skip

            with ThreadPoolExecutor(max_workers=procs) as pool:
                list(pool.map(one, range(n)))
        seconds = time.perf_counter() - start
        curves, version = {}, None
        for k in range(n):
            directory = work / f"r{k:05d}"
            log = (directory / "log.out").read_text() if (directory / "log.out").exists() else ""
            found = re.search(r"SimSS version (\d+(?:\.\d+)+)", log)
            version = version or (found.group(1) if found else None)
            if (directory / "JV.dat").exists():
                data = np.atleast_1d(np.genfromtxt(directory / "JV.dat", names=True))
                J = np.full(V.shape, np.nan)
                where = np.rint((data["Vext"] - V[0]) / (V[1] - V[0])).astype(int)
                keep = (where >= 0) & (where < V.size)
                J[where[keep]] = data["Jext"][keep]
                curves[k] = J
    finally:
        shutil.rmtree(work, ignore_errors=True)
    row = {"solver": "simss", "device": "cpu", "procs": procs, "runs": n, "seconds": seconds,
           "runs_per_s": n / seconds, "version": version,
           "how": f"docker {docker}" if docker else f"native {plain}, {fixed}",
           "failed_runs": n - len(curves),
           "skipped_voltages": float(np.mean([np.isnan(c).mean() for c in curves.values()]))
           if curves else None}  # fmt: skip
    print(f"simss {procs} processes: {row['runs_per_s']:.2f} runs/s")
    return row, curves


def accuracy(curves: dict[int, np.ndarray]) -> dict:
    from bench import variants

    from jaxdd.batch import prepare, solve_batch

    keys = sorted(curves)[:ACCURACY_DEVICES]
    devices = variants(max(keys) + 1, SEED)
    J, ok, _ = solve_batch(prepare([devices[k] for k in keys]), V, bernoulli_name="simss")
    J = np.where(np.asarray(ok), np.asarray(J), np.nan)
    gaps = []
    for row, k in enumerate(keys):
        scale = max(np.nanmax(np.abs(curves[k])), 1.0)
        gaps.append(float(np.nanmax(np.abs(J[row] - curves[k])) / scale))
    gaps = np.array(gaps)
    return {"devices": len(keys), "bernoulli": "simss", "tolerance": ACCURACY_TOL,
            "median": float(np.median(gaps)), "max": float(gaps.max()),
            "beyond_tolerance": int((gaps > ACCURACY_TOL).sum())}  # fmt: skip


# ------------------------------------------------------------------------------- main
def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="the machine's name in the plot")
    parser.add_argument("--skip", default="", help="parts to leave out: gpu,cpu,simss,accuracy")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--gpu-batches", default="4096,16384")
    parser.add_argument("--gpu-unique", type=int, default=2048)
    parser.add_argument("--cpu-procs", type=int, default=os.cpu_count())
    parser.add_argument("--cpu-batch", type=int, default=32)
    parser.add_argument("--simss", default=None, help="SimSS's binary (on PATH or a path)")
    parser.add_argument("--simss-fixed", default=None, help="the patched binary for frozen ions")
    parser.add_argument("--simss-docker", default=None, help="image with both SimSS binaries")
    parser.add_argument("--docker-cmd", default="docker", help='e.g. "wsl -e docker" on Windows')
    parser.add_argument("--simss-procs", type=int, default=os.cpu_count())
    parser.add_argument("--simss-runs", type=int, default=None, help="default 4 per process")
    args = parser.parse_args(argv)
    skip = set(filter(None, args.skip.split(",")))

    import jax

    report = {
        "label": args.label,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "machine": {"cpu": _cpu_model(), "logical_cores": os.cpu_count(),
                    "memory_gb": _memory_gb(), "os": platform.platform()},
        "versions": {"jaxdd": _jaxdd_commit(), "jax": jax.__version__,
                     "python": platform.python_version()},
        "workload": {"cases": "validation/bench.py CASES, random traps, ions and doping",
                     "seed": SEED, "voltages": int(V.size), "run": "one J-V curve"},
        "results": [],
    }  # fmt: skip
    if "gpu" not in skip and jax.default_backend() != "cpu":
        report["machine"]["gpu"] = jax.devices()[0].device_kind
        batches = [int(b) for b in args.gpu_batches.split(",")]
        report["results"] += measure_gpu(batches, args.gpu_unique, args.repeats)
    if "cpu" not in skip:
        report["results"].append(measure_cpu(args.cpu_procs, args.cpu_batch, args.repeats))
    if "simss" not in skip and (args.simss or args.simss_docker):
        plain = args.simss or "simss"
        fixed = args.simss_fixed or ("simss_fixed_ions" if args.simss_docker else plain)
        n = args.simss_runs or max(16, 4 * args.simss_procs)
        row, curves = measure_simss(n, args.simss_procs, plain, fixed, args.simss_docker,
                                    args.docker_cmd.split())  # fmt: skip
        report["results"].append(row)
        report["versions"]["simss"] = row["version"]
        if curves and "accuracy" not in skip:
            report["accuracy"] = accuracy(curves)
            print(f"accuracy: {report['accuracy']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{_slug(args.label)}.json"
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"written {path.relative_to(ROOT)}; now: python benchmarks/plot.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
