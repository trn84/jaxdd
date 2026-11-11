#!/usr/bin/env bash
# jaxdd's benchmarks on a cloud GPU pod (RunPod or alike): a container with a current NVIDIA
# driver, where current JAX runs directly, without Apptainer (benchmarks/apptainer.sh).
#
# Makes its own venv (.venv, Python 3.12 from uv, current jax[cuda12]) on the first start,
# whatever Python the image has; installs uv if missing. Runs the steps in the background, so
# a closed web terminal or SSH session does not stop them, and follows the log.
#
# Once, in the pod's terminal (/workspace survives a stop of the pod, the rest does not):
#   cd /workspace && git clone https://github.com/trn84/jaxdd.git && cd jaxdd
#
# Each time:
#   cd /workspace/jaxdd && git pull
#   benchmarks/runpod.sh                          # every step, about 25 min on an A100
#   STEPS="breakdown bench scan" benchmarks/runpod.sh
# Ctrl+C ends only the following of the log; the run goes on. To follow it again:
#   tail -F benchmarks/runpod.log
#
# Steps (STEPS, in this order by default):
#   info       the GPU, its driver, JAX and its devices, the pod's CPU cores
#   tests      the test suite (a broken install fails here, in a minute)
#   breakdown  validation/breakdown.py at batch 4096 and 16384: one Newton step, part by part
#   bench      validation/bench.py 65536 16384: curves per second
#   scan       breakdown.py 16384 and bench.py 65536 16384 again with JAXDD_SWEEP=scan: the
#              block sweep as a scan instead of the Pallas kernel, on the same card
#   hard       validation/hard.py 16384 16384: a hard workload (frozen ions over a wide range,
#              light and dark), two passes against one, the Newton steps a batch pays
#   run        benchmarks/run.py: GPU at batch 4096 and 16384, jaxdd on the pod's CPU cores;
#              writes benchmarks/results/<label>.json and prints it into the log
#
# Settings (environment): STEPS; LABEL, the machine's name in the plot (default: the GPU's
# name and "pod"); CPU_PROCS, the processes of the CPU part (default: the cores the pod's
# cgroup grants, not the host's, which os.cpu_count() would give); VENV. Everything is
# appended to benchmarks/runpod.log (not committed).
set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT="$PWD"
VENV="${VENV:-$ROOT/.venv}"
LOG="$ROOT/benchmarks/runpod.log"
STEPS="${STEPS:-info tests breakdown bench scan hard run}"

say() { printf '\n== %s  %s\n' "$(date '+%F %T')" "$*"; }

# The CPU cores this container may use: a pod sees every core of its host (nproc too), but
# its cgroup's quota grants only some of them.
pod_cores() {
    local quota period n
    if [ -r /sys/fs/cgroup/cpu.max ]; then                        # cgroup v2
        read -r quota period < /sys/fs/cgroup/cpu.max
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then         # cgroup v1
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    fi
    n=$(nproc)
    if [ -n "${quota:-}" ] && [ "$quota" != max ] && [ "$quota" -gt 0 ]; then
        local granted=$(( (quota + period - 1) / period ))
        [ "$granted" -lt "$n" ] && n=$granted
    fi
    echo "$n"
}

# ------------------------------------------------------------------ the steps, in the background
if [ "${1:-}" = "--inside" ]; then
    py="$VENV/bin/python"
    for step in $STEPS; do
        say "$step"
        case "$step" in
            info)
                nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm \
                    --format=csv || true
                "$py" -c "import jax; print('JAX', jax.__version__, jax.devices())"
                echo "$CPU_PROCS cores for this pod (the host has $(nproc));" \
                     "$(awk '/MemTotal/ {printf "%.0f GB", $2/1e6}' /proc/meminfo) host memory"
                ;;
            tests)
                "$py" -m pytest -q
                ;;
            breakdown)
                (cd validation && "$py" breakdown.py 4096 && "$py" breakdown.py 16384)
                ;;
            bench)
                "$py" validation/bench.py 65536 16384
                ;;
            scan)
                (cd validation && JAXDD_SWEEP=scan "$py" breakdown.py 16384) &&
                    JAXDD_SWEEP=scan "$py" validation/bench.py 65536 16384
                ;;
            hard)
                "$py" validation/hard.py 16384 16384
                ;;
            run)
                # No SimSS on the pod: jaxdd on the GPU and on the pod's cores only.
                marker="$ROOT/benchmarks/.runpod-start"
                touch "$marker"
                rc=0
                "$py" benchmarks/run.py --label "$LABEL" --skip simss,accuracy \
                    --cpu-procs "$CPU_PROCS" || rc=$?
                find "$ROOT/benchmarks/results" -name '*.json' -newer "$marker" \
                    -exec sh -c 'echo "--- $1"; cat "$1"; echo' _ {} \;
                rm -f "$marker"
                (exit $rc)
                ;;
            *)
                echo "unknown step $step (info tests breakdown bench scan hard run)"
                ;;
        esac || echo "!! step $step failed (exit $?); going on"
    done
    say "done"
    exit 0
fi

# ------------------------------------------------------------------ setting up, starting
command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: a pod with a GPU?"; exit 1; }
if pgrep -f "benchmarks/runpod.sh --inside" >/dev/null; then
    echo "a benchmark is already running (pgrep -af runpod.sh); two would share the GPU"
    exit 1
fi

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
if [ ! -x "$UV" ]; then
    echo "installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi
# /workspace is often another filesystem than uv's cache: copy instead of failing hard links.
export UV_LINK_MODE=copy
if [ ! -x "$VENV/bin/python" ]; then
    echo "making $VENV (Python 3.12, current jax[cuda12]); a few minutes"
    "$UV" venv --python 3.12 --python-preference only-managed "$VENV"
    "$UV" pip install --python "$VENV/bin/python" -e "$ROOT[dev,bench]" "jax[cuda12]"
fi

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
LABEL="${LABEL:-${GPU#NVIDIA } pod}"
CPU_PROCS="${CPU_PROCS:-$(pod_cores)}"
COMMIT="$(git rev-parse --short=12 HEAD)$(git diff --quiet || echo ' with local changes')"
export STEPS LABEL CPU_PROCS

start=$(( $(stat -c %s "$LOG" 2>/dev/null || echo 0) + 1 ))
{
    say "start: steps $STEPS, label \"$LABEL\", $CPU_PROCS CPU processes, JAXDD_SWEEP ${JAXDD_SWEEP:-auto}"
    echo "jaxdd $COMMIT"
} >> "$LOG"
echo "starting the steps \"$STEPS\"; log $LOG"
nohup "$ROOT/benchmarks/runpod.sh" --inside >> "$LOG" 2>&1 < /dev/null &
pid=$!
echo "pid $pid; Ctrl+C ends only the following, not the run"
sleep 2
tail -c +"$start" -F --pid="$pid" "$LOG" 2>/dev/null
