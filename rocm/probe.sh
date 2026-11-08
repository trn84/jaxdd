#!/usr/bin/env bash
# M8, inside the ROCm JAX container (rocm/mi50.sh starts it): does jaxdd run on the MI50, on
# every card, how large a batch fits in 16 GB, and how fast is it? Every step runs even when
# one before failed, so that one log shows all of it.
set -uo pipefail
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
PY="${PY:-python3}"
#: Batch sizes to try on card 0; SKIP_BENCH=1 leaves out benchmarks/run.py (a dry run on a CPU).
BATCHES="${BATCHES:-1024 2048 4096 8192}"

say() { printf '\n== %s  %s\n' "$(date '+%F %T')" "$*"; }

say "versions"
$PY --version
$PY -c 'import jax, jaxlib, numpy; print("jax", jax.__version__, "jaxlib", jaxlib.__version__, "numpy", numpy.__version__)'
cat "$ROCM_PATH/.info/version" 2>/dev/null || ls "$ROCM_PATH"/..

say "GPUs"
rocminfo 2>/dev/null | grep -E "Marketing Name|Name: +gfx" | sort | uniq -c
CARDS=$(rocminfo 2>/dev/null | grep -c "Name: *gfx906")
echo "gfx906 agents: $CARDS"
$PY -c 'import jax; print(jax.default_backend(), jax.devices())'

say "install jaxdd (no dependencies: the image's JAX and numpy stay)"
$PY -m pip install -q --no-deps -e . && $PY -m pip install -q pytest

say "smoke: jit, float64, one jaxdd batch on the GPU"
$PY - <<'EOF'
import time

import jax
import jax.numpy as jnp
import numpy as np

print("softmax under jit:", jax.jit(jax.nn.softmax)(jnp.ones(4)))
from jaxdd.batch import prepare, solve_batch  # turns on float64

x = jnp.ones(3, dtype=jnp.float64)
print("float64 on", x.devices() if hasattr(x, "devices") else x.device(), x.dtype)
import sys

sys.path.insert(0, "validation")
from bench import V, variants

batch = prepare(variants(8))
start = time.perf_counter()
J, ok, its = solve_batch(batch, V)
J.block_until_ready()
print(f"8 runs with compilation: {time.perf_counter() - start:.1f} s, converged {bool(np.asarray(ok).all())}")
EOF

say "tests on the GPU"
$PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -5

say "every card alone (the same 8 runs)"
for i in $(seq 0 $((CARDS > 0 ? CARDS - 1 : 0))); do
    HIP_VISIBLE_DEVICES=$i $PY -c '
import sys, time
sys.path.insert(0, "validation")
import numpy as np
from bench import V, variants
from jaxdd.batch import prepare, solve_batch
J, ok, its = solve_batch(prepare(variants(8)), V); J.block_until_ready()
print("card '"$i"':", "converged" if bool(np.asarray(ok).all()) else "NOT converged", float(np.asarray(J)[0, 0]))
' 2>&1 | tail -1
done

say "batch sizes on card 0, up to its 16 GB"
for b in $BATCHES; do
    HIP_VISIBLE_DEVICES=0 timeout 1800 $PY validation/bench.py $((2 * b)) $b 2>&1 | tail -2 \
        || echo "batch $b: failed (out of memory?)"
done

say "benchmark (benchmarks/run.py): one card and this machine's CPU"
[ "${SKIP_BENCH:-0}" = 1 ] || HIP_VISIBLE_DEVICES=0 $PY benchmarks/run.py --label "MI50 (one card)" --gpu-batches 2048,4096,8192 \
    --skip simss 2>&1 | tail -8
