#!/usr/bin/env bash
# jaxdd's benchmarks on a GPU host too old for current JAX, inside an Apptainer container.
#
# A host with an old glibc (2.17) and an old NVIDIA driver (530, CUDA 12.1) takes no current
# JAX wheel, and the newest JAX that installs there runs out of memory on that driver
# (docs/hardware.md). In a container with a current Ubuntu, current JAX runs on the host's
# driver. Started on the host; it starts the container in the background, so a closed terminal
# does not stop it, and follows the log. On a Slurm-run hub everything ends with the job,
# nohup included.
#
# Once: uv in ~/.local/bin and the repository:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   git clone https://github.com/trn84/jaxdd.git ~/jaxdd
# The image (~100 MB) and the container's venv (.venvct) are made on the first start if missing.
#
# Each time:
#   cd ~/jaxdd && git pull
#   benchmarks/apptainer.sh                       # every step, about 20 min on an A100
#   STEPS="breakdown bench" benchmarks/apptainer.sh
# Ctrl+C ends only the following of the log; the run goes on. To follow it again:
#   tail -F benchmarks/apptainer.log
#
# Steps (STEPS, in this order by default):
#   info       the GPU, JAX and its devices
#   tests      the test suite (a broken install fails here, in a minute)
#   breakdown  validation/breakdown.py at batch 4096 and 16384: one Newton step, part by part
#   bench      validation/bench.py 65536 16384: curves per second
#   hard       validation/hard.py 16384 16384: a hard workload (frozen ions over a wide range,
#              light and dark), two passes against one, the Newton steps a batch pays
#   run        benchmarks/run.py: GPU at batch 4096 and 16384, jaxdd on this job's CPU cores;
#              writes benchmarks/results/<label>.json for benchmarks/plot.py
#   chunk      validation/chunk.py on the batch file CHUNK (a real workload built elsewhere and
#              copied here; not in the default steps)
#   sample     validation/sample_measured.py on each of SAMPLES (default "02Cl 06Cl 01Cl"), with
#              SAMPLE_ARGS (default "--walkers 128 --steps 3000"): a posterior of the measured
#              cells' parameters; each chain to benchmarks/samples/chain_<sample>.npz, saved every
#              100 steps, so a session that ends keeps what is done. Not in the default steps:
#                STEPS=sample benchmarks/apptainer.sh
#
# Settings (environment): STEPS; CHUNK, the batch file of the chunk step; SAMPLES and SAMPLE_ARGS
# for the sample step; LABEL, the machine's name in the plot (default: the GPU's
# name and "apptainer"); SIF, the image file (default cuda129.sif in the repository); IMAGE,
# where it is pulled from. Everything is appended to benchmarks/apptainer.log (not committed).
set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT="$PWD"
SIF="${SIF:-$ROOT/cuda129.sif}"
IMAGE="${IMAGE:-docker://nvidia/cuda:12.9.1-base-ubuntu24.04}"
VENV="${VENV:-$ROOT/.venvct}"
LOG="$ROOT/benchmarks/apptainer.log"
STEPS="${STEPS:-info tests breakdown bench hard run}"

say() { printf '\n== %s  %s\n' "$(date '+%F %T')" "$*"; }

# ------------------------------------------------------------------ inside the container
if [ "${1:-}" = "--inside" ]; then
    py="$VENV/bin/python"
    for step in $STEPS; do
        say "$step"
        case "$step" in
            info)
                nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
                "$py" -c "import jax; print('JAX', jax.__version__, jax.devices())"
                echo "$(nproc) cores for this job"
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
            hard)
                "$py" validation/hard.py 16384 16384
                ;;
            run)
                # No SimSS in the container: jaxdd on the GPU and on this job's cores only.
                "$py" benchmarks/run.py --label "$LABEL" --skip simss,accuracy \
                    --cpu-procs "$(nproc)"
                ;;
            chunk)
                "$py" validation/chunk.py "$CHUNK"
                ;;
            sample)
                mkdir -p benchmarks/samples
                for s in ${SAMPLES:-02Cl 06Cl 01Cl}; do
                    say "sample $s"
                    # shellcheck disable=SC2086
                    "$py" -u validation/sample_measured.py "$s" \
                        ${SAMPLE_ARGS:---walkers 128 --steps 3000} \
                        --out "benchmarks/samples/chain_$s.npz" ||
                        echo "!! sample $s failed (exit $?); going on"
                done
                ;;
            *)
                echo "unknown step $step (info tests breakdown bench hard run chunk sample)"
                ;;
        esac || echo "!! step $step failed (exit $?); going on"
    done
    say "done"
    exit 0
fi

# ------------------------------------------------------------------ on the host
if ! command -v apptainer >/dev/null; then
    # `module` is a shell function of the login shell, not always defined in a script.
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true
    module load apptainer 2>/dev/null || true
fi
command -v apptainer >/dev/null || { echo "no apptainer: module load apptainer, then again"; exit 1; }
UV="$HOME/.local/bin/uv"
[ -x "$UV" ] || { echo "no uv in ~/.local/bin: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
if pgrep -u "$(id -un)" -f "benchmarks/apptainer.sh --inside" >/dev/null; then
    echo "a benchmark is already running (pgrep -af apptainer.sh); two would share the GPU"
    exit 1
fi
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "this job has $(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || echo '?') left" \
         "(every default step: about 20 min on an A100; the sample step hours)"
fi

# The host's LD_LIBRARY_PATH stays out, as do the image's CUDA compat libraries, which need a
# newer driver (`cuInit 803` on 530). The container's PATH gets uv.
unset LD_LIBRARY_PATH
export APPTAINERENV_PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

if [ ! -f "$SIF" ]; then
    echo "pulling $IMAGE into $SIF"
    apptainer pull "$SIF" "$IMAGE"
fi
if [ ! -x "$VENV/bin/python" ]; then
    echo "making $VENV in the container (Python 3.12, current jax[cuda12]); a few minutes"
    apptainer -s exec --nv "$SIF" bash -c "
        '$UV' venv --python 3.12 --python-preference only-managed '$VENV' &&
        '$UV' pip install --python '$VENV/bin/python' -e '$ROOT[dev,bench]' 'jax[cuda12]'"
fi

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
LABEL="${LABEL:-${GPU#NVIDIA } apptainer}"
COMMIT="$(git rev-parse --short=12 HEAD)$(git diff --quiet || echo ' with local changes')"
export APPTAINERENV_STEPS="$STEPS" APPTAINERENV_LABEL="$LABEL" APPTAINERENV_JAXDD_COMMIT="$COMMIT"
[ -n "${SAMPLES:-}" ] && export APPTAINERENV_SAMPLES="$SAMPLES"
[ -n "${SAMPLE_ARGS:-}" ] && export APPTAINERENV_SAMPLE_ARGS="$SAMPLE_ARGS"
if [[ " $STEPS " == *" chunk "* ]]; then
    [ -f "${CHUNK:-}" ] || { echo "the chunk step needs CHUNK=<batch file>"; exit 1; }
    export APPTAINERENV_CHUNK="$(realpath "$CHUNK")"
fi

start=$(( $(stat -c %s "$LOG" 2>/dev/null || echo 0) + 1 ))
{
    say "start: steps $STEPS, label \"$LABEL\", JAXDD_SWEEP ${JAXDD_SWEEP:-auto}"
    echo "jaxdd $COMMIT, image $SIF"
} >> "$LOG"
echo "starting the steps \"$STEPS\" in $SIF; log $LOG"
# -s hides apptainer's harmless "gocryptfs not found".
nohup apptainer -s exec --nv "$SIF" "$ROOT/benchmarks/apptainer.sh" --inside \
    >> "$LOG" 2>&1 < /dev/null &
pid=$!
echo "pid $pid; Ctrl+C ends only the following, not the run"
sleep 5
kill -0 "$pid" 2>/dev/null || { echo "it stopped at once:"; tail -20 "$LOG"; exit 1; }
tail -c +"$start" -F --pid="$pid" "$LOG" 2>/dev/null
