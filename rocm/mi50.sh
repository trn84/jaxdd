#!/usr/bin/env bash
# jaxdd on AMD Instinct MI50 (gfx906), in a container whose JAX still runs on it.
#
#   git clone https://github.com/trn84/jaxdd.git && cd jaxdd
#   rocm/mi50.sh                     # IMAGE=... rocm/mi50.sh to try another ROCm (below)
#
# gfx906 left ROCm's supported list with 6.3, and no current JAX build targets it. What works,
# by jax-ml/jax#27600 (2025, on an MI50): rocm/jax-community:rocm6.0.3-jax0.4.31-py3.11.10 with
# ROCM_PATH=/opt/rocm (without it the PJRT client segfaulted). From JAX 0.5 on, XLA compiles
# fusions through Triton, which refuses gfx906. jaxdd needs jax >= 0.4.30: its tests pass on JAX
# 0.4.31 with numpy 1.26 (CPU, 2026-11-08), its curves within 9e-7 of the current JAX's.
# Fallbacks with the same JAX: rocm6.1.3-jax0.4.31-py3.11.10, rocm6.2.3-jax0.4.31-py3.11.10.
#
# Needs on the host: the amdgpu kernel driver (/dev/kfd), Docker, the user in the video and
# render groups. Everything is logged to rocm/mi50_probe.log (appended to, not committed).
set -uo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-rocm/jax-community:rocm6.0.3-jax0.4.31-py3.11.10}"
LOG=rocm/mi50_probe.log
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n== %s  %s\n' "$(date '+%F %T')" "$*"; }

say "host"
echo "jaxdd $(git rev-parse --short HEAD), image $IMAGE"
uname -r
ls -l /dev/kfd /dev/dri 2>&1 | head -12
id
grep -m1 'model name' /proc/cpuinfo || true
echo "$(nproc) threads"; free -g | head -2
command -v rocm-smi >/dev/null && rocm-smi --showproductname --showmeminfo vram 2>&1 | head -40
docker version --format 'docker {{.Server.Version}}' || { echo "no Docker"; exit 1; }
[ -e /dev/kfd ] || { echo "no /dev/kfd: the amdgpu kernel driver is not loaded"; exit 1; }

say "image"
docker pull -q "$IMAGE"

# The container needs the render nodes' groups by number: names differ between host and image.
groups_args=()
for g in video render; do
    gid=$(getent group "$g" | cut -d: -f3) && [ -n "$gid" ] && groups_args+=(--group-add "$gid")
done

say "probe (in the container)"
docker run --rm --device=/dev/kfd --device=/dev/dri "${groups_args[@]}" --ipc=host \
    --security-opt seccomp=unconfined -e ROCM_PATH=/opt/rocm -e PYTHONUNBUFFERED=1 \
    -v "$PWD":/jaxdd -w /jaxdd "$IMAGE" bash rocm/probe.sh

say "done: send rocm/mi50_probe.log, and benchmarks/results/*.json if the benchmark ran"
