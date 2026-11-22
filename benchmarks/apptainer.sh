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
# The image (~100 MB) and the container's venv (.venvct) are made on the first start if missing;
# every start installs into the venv what pyproject.toml asks for and it lacks.
#
# Each time:
#   cd ~/jaxdd && git pull
#   benchmarks/apptainer.sh                       # the default steps, about 20 min on an A100
#   STEPS=all benchmarks/apptainer.sh             # all of a full measurement, about 40 min
#   STEPS="breakdown bench" benchmarks/apptainer.sh
# Ctrl+C ends only the following of the log; the run goes on. To follow it again:
#   tail -F benchmarks/apptainer.log
#
# solve_batch splits a batch across every GPU the job sees, so on a host with several the default
# steps measure them together; one GPU's numbers: CUDA_VISIBLE_DEVICES=0 benchmarks/apptainer.sh.
#
# Steps (STEPS, in this order by default):
#   info       the GPU, JAX and its devices
#   tests      the test suite (a broken install fails here, in a minute); in parallel, a worker
#              per core of the job, but in one process where the GPUs are exclusive to one
#              (compute mode Exclusive_Process): 221 tests in 34 min on an A100 there, which the
#              durations below do not include
#   breakdown  validation/breakdown.py at batch 4096 and 16384: one Newton step, part by part
#   bench      validation/bench.py 65536 16384: curves per second
#   hard       validation/hard.py 16384 16384: a hard workload (frozen ions over a wide range,
#              light and dark), two passes against one, the Newton steps a batch pays
#   steric     validation/hard.py 16384 16384 N_MAX for each of ${STERIC_ARGS:-1.21e28 2e27 1e26}:
#              the hard workload with the absorber's ions bounded by that many sites (m⁻³), its
#              curves against Boltzmann's; not in the default steps (a few minutes on an A100)
#   slots      validation/slots.py 16384 4096: devices with two mobile ion regions per species
#              against one, through the sweep kernel and the scan (about 5 min)
#   gpus       validation/bench.py on one GPU (CUDA_VISIBLE_DEVICES=0) and on all the job's GPUs,
#              the batch split across them (solve_batch's devices): 16384 per GPU, and 16384 in
#              all; not in the default steps (a host with several GPUs)
#   grid       validation/grid.py ${GRID_ARGS:-4096 4096}: M4, the hard devices in the
#              light and the dark on SIMsalabim's grid and the graded one at 200 to 3200 points
#              (about 15 min per grid on one A100), against the graded one at 3200; saved per
#              grid and size and resumed (validation/grid_<N>.npz); not in the default steps.
#              GRID_ARGS="4096 4096 1600,3200 graded" for some sizes or one grid only
#   run        benchmarks/run.py: GPU at batch 4096 and 16384 per GPU, jaxdd on this job's CPU cores;
#              writes benchmarks/results/<label>.json for benchmarks/plot.py
#   chunk      validation/chunk.py on the batch file CHUNK (a real workload built elsewhere and
#              copied here; not in the default steps)
#   transient  validation/transient_bench.py ${TRANSIENT_ARGS:-16384 4096}: Stage C's speed,
#              transients per second with traps (3×3), with mobile ions (5×5) and at the open
#              circuit, the Newton steps per time step and one Newton step's cost; not in the
#              default steps (about 10 min on an A100)
#   sample     validation/sample_measured.py on each of SAMPLES (default "02Cl 06Cl 01Cl"), with
#              SAMPLE_ARGS (default "--walkers 128 --steps 3000"): a posterior of the measured
#              cells' parameters; each chain to benchmarks/samples/chain_<sample>.npz, saved every
#              100 steps, so a session that ends keeps what is done. Not in the default steps:
#                STEPS=sample benchmarks/apptainer.sh
#              with UPLOAD (below) the results go to a shared folder as they come:
#                STEPS=sample UPLOAD=https://HOST/s/TOKEN benchmarks/apptainer.sh
#   trpl       validation/trpl_sample.py --outcoupling: the posterior of the joint TRPL fit with
#              outcoupling near the glass, one chain per seed of TRPL_SEEDS (default "0 1"), the
#              chains side by side, each on a GPU of its own where the job has as many, else
#              sharing them, each process then with TRPL_MEM (default 0.45) of its memory;
#              with TRPL_ARGS (default "--walkers 64 --steps 4000 --every 50"); each chain to
#              benchmarks/samples/trpl_out_chain_s<seed>.npz, saved every 50 steps and resumed by
#              the next start (`--resume`), so a job that ends loses at most 50 steps. Hours on
#              an A100; not in the default steps:
#                STEPS=trpl benchmarks/apptainer.sh
#              `--neutral` in TRPL_ARGS samples the fit with saturable SnO2 interface states
#              instead (26 parameters; trpl_neutral_chain_s<seed>.npz):
#                STEPS=trpl TRPL_ARGS="--neutral --walkers 256 --steps 1000 --every 25" ...
#              and with the SnO2 decays' background taken off (`--neutral --background`,
#              trpl_bg_ec3.9_chain_s<seed>.npz; the fit refitted once before the chains on a
#              clone without validation/trpl_joint_..._ec3.9_bg.json):
#                STEPS=trpl TRPL_ARGS="--neutral --background --walkers 256 --steps 3000 \
#                    --every 25" UPLOAD=https://HOST/s/TOKEN benchmarks/apptainer.sh
#              Before the chains, the best fit once (trpl_sample.py --fit-only).
#
# STEPS=all runs every step of a full measurement: info tests breakdown bench hard steric slots
# transient run (about 40 min on an A100; not gpus, grid, chunk, sample, trpl).
#
# Settings (environment): STEPS; CHUNK, the batch file of the chunk step; SAMPLES and SAMPLE_ARGS
# for the sample step; TRPL_SEEDS, TRPL_ARGS and TRPL_MEM for the trpl step; UPLOAD, a public
# share link of a Nextcloud or ownCloud folder with upload rights (https://HOST/s/TOKEN): the log,
# the chains of the sample and trpl steps and the TRPL fits and summaries (validation/trpl_*.json)
# are uploaded to it every UPLOAD_EVERY seconds
# (default 600) while the run lives, and once at its end; LABEL, the machine's name in the plot
# (default: the GPU's name and "apptainer"); SIF, the image file (default cuda129.sif in the
# repository); IMAGE, where it is pulled from. Everything is appended to benchmarks/apptainer.log
# (not committed).
set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT="$PWD"
SIF="${SIF:-$ROOT/cuda129.sif}"
IMAGE="${IMAGE:-docker://nvidia/cuda:12.9.1-base-ubuntu24.04}"
VENV="${VENV:-$ROOT/.venvct}"
LOG="$ROOT/benchmarks/apptainer.log"
STEPS="${STEPS:-info tests breakdown bench hard slots run}"
[ "$STEPS" = "all" ] && STEPS="info tests breakdown bench hard steric slots transient run"

say() { printf '\n== %s  %s\n' "$(date '+%F %T')" "$*"; }

# ------------------------------------------------------------------ the uploads (on the host)
upload() {
    local base="${UPLOAD%%/s/*}" token="${UPLOAD##*/s/}" f
    token="${token%%[/?#]*}"
    for f in "$LOG" "$ROOT"/benchmarks/samples/*.npz "$ROOT"/validation/trpl_*.json; do
        [ -f "$f" ] || continue
        curl -sS -f -u "$token:" -T "$f" "$base/public.php/webdav/$(basename "$f")" ||
            echo "!! upload of $(basename "$f") failed"
    done
}
if [ "${1:-}" = "--upload" ]; then
    pid="$2"
    while kill -0 "$pid" 2>/dev/null; do
        for _ in $(seq $(( ${UPLOAD_EVERY:-600} / 10 ))); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 10
        done
        upload
    done
    say "uploaded the log and the chains"
    upload
    exit 0
fi

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
                # A GPU in exclusive-process mode takes one process: every parallel worker but
                # one fails with CUDA_ERROR_DEVICE_UNAVAILABLE, so the tests run in one there.
                # Otherwise a worker per core of this job (-n auto counts the host's).
                if nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null |
                        grep -qi exclusive; then
                    echo "the GPUs are exclusive to one process: the tests in one"
                    "$py" -m pytest -q -n 0
                else
                    "$py" -m pytest -q -n "$(nproc)"
                fi
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
            steric)
                for n in ${STERIC_ARGS:-1.21e28 2e27 1e26}; do
                    say "steric N_max = $n"
                    "$py" validation/hard.py 16384 16384 "$n" || echo "!! steric $n failed (exit $?)"
                done
                ;;
            slots)
                "$py" validation/slots.py 16384 4096
                ;;
            gpus)
                CUDA_VISIBLE_DEVICES=0 "$py" validation/bench.py 65536 16384
                "$py" validation/bench.py 65536 32768
                "$py" validation/bench.py 65536 16384
                ;;
            grid)
                # shellcheck disable=SC2086
                (cd validation && "$py" grid.py ${GRID_ARGS:-4096 4096})
                ;;
            run)
                # No SimSS in the container: jaxdd on the GPU and on this job's cores only.
                "$py" benchmarks/run.py --label "$LABEL" --skip simss,accuracy \
                    --cpu-procs "$(nproc)"
                ;;
            chunk)
                "$py" validation/chunk.py "$CHUNK"
                ;;
            transient)
                # shellcheck disable=SC2086
                "$py" validation/transient_bench.py ${TRANSIENT_ARGS:-16384 4096}
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
            trpl)
                mkdir -p benchmarks/samples
                pids=()
                seeds=(${TRPL_SEEDS:-0 1})
                gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
                [ "$gpus" -ge 1 ] || gpus=1
                mem=0.75  # JAX's own share where a chain has its GPU alone
                [ "$gpus" -lt "${#seeds[@]}" ] && mem="${TRPL_MEM:-0.45}"
                # The best fit once before the chains, which would each refit it where it is
                # missing (--background on a fresh clone).
                # shellcheck disable=SC2086
                (cd validation && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
                    "$py" -u trpl_sample.py --outcoupling ${TRPL_ARGS:-} --fit-only 2>&1 |
                    grep --line-buffered -v Warning) ||
                    echo "!! the trpl fit failed (exit $?)"
                for k in "${!seeds[@]}"; do
                    seed=${seeds[$k]}
                    echo "chain of seed $seed on GPU $(( k % gpus )) with $mem of its memory"
                    # shellcheck disable=SC2086
                    (cd validation && CUDA_VISIBLE_DEVICES=$(( k % gpus )) \
                        XLA_PYTHON_CLIENT_MEM_FRACTION="$mem" \
                        "$py" -u trpl_sample.py --outcoupling --seed "$seed" --resume \
                        ${TRPL_ARGS:---walkers 64 --steps 4000 --every 50} 2>&1 |
                        grep --line-buffered -v Warning | sed -u "s/^/[seed $seed] /") &
                    pids+=($!)
                done
                for pid in "${pids[@]}"; do
                    wait "$pid" || echo "!! a trpl chain failed (exit $?)"
                done
                ;;
            *)
                echo "unknown step $step (info tests breakdown bench hard steric slots gpus grid run chunk transient sample trpl)"
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
         "(every default step: about 20 min on an A100; the sample and trpl steps hours)"
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
    apptainer -s exec --nv "$SIF" \
        "$UV" venv --python 3.12 --python-preference only-managed "$VENV"
fi
# Every start: what pyproject.toml asks for now (a venv made before pytest-xdist was in the dev
# extra failed the tests step with "unrecognized arguments: -n"); what is there stays, JAX too.
apptainer -s exec --nv "$SIF" \
    "$UV" pip install -q --python "$VENV/bin/python" -e "$ROOT[dev,bench]" 'jax[cuda12]'

GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
LABEL="${LABEL:-${GPU#NVIDIA } apptainer}"
COMMIT="$(git rev-parse --short=12 HEAD)$(git diff --quiet || echo ' with local changes')"
export APPTAINERENV_STEPS="$STEPS" APPTAINERENV_LABEL="$LABEL" APPTAINERENV_JAXDD_COMMIT="$COMMIT"
[ -n "${SAMPLES:-}" ] && export APPTAINERENV_SAMPLES="$SAMPLES"
[ -n "${SAMPLE_ARGS:-}" ] && export APPTAINERENV_SAMPLE_ARGS="$SAMPLE_ARGS"
[ -n "${GRID_ARGS:-}" ] && export APPTAINERENV_GRID_ARGS="$GRID_ARGS"
[ -n "${TRANSIENT_ARGS:-}" ] && export APPTAINERENV_TRANSIENT_ARGS="$TRANSIENT_ARGS"
[ -n "${TRPL_SEEDS:-}" ] && export APPTAINERENV_TRPL_SEEDS="$TRPL_SEEDS"
[ -n "${TRPL_ARGS:-}" ] && export APPTAINERENV_TRPL_ARGS="$TRPL_ARGS"
[ -n "${TRPL_MEM:-}" ] && export APPTAINERENV_TRPL_MEM="$TRPL_MEM"
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
if [ -n "${UPLOAD:-}" ]; then
    nohup "$ROOT/benchmarks/apptainer.sh" --upload "$pid" >> "$LOG" 2>&1 < /dev/null &
    echo "uploading the log and the chains every ${UPLOAD_EVERY:-600} s and at the end"
fi
sleep 5
kill -0 "$pid" 2>/dev/null || { echo "it stopped at once:"; tail -20 "$LOG"; exit 1; }
tail -c +"$start" -F --pid="$pid" "$LOG" 2>/dev/null
