"""`benchmarks/run.py` gives SimSS the very devices jaxdd solves: its input writer, checked."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "validation"))

pytest.importorskip("bench")
from bench import variant_setups  # noqa: E402
from run import write_simss_inputs  # noqa: E402

from jaxdd.simss_input import read_setup  # noqa: E402


@pytest.mark.parametrize("k", range(3))
def test_simss_reads_back_the_variant(tmp_path, k):
    """Every value of the variant, and only its changes, in SimSS's files."""
    setup = variant_setups(3)[k]
    write_simss_inputs(setup, tmp_path / "run")
    back = read_setup(tmp_path / "run" / "simulation_setup.txt")
    assert back.raw == setup.raw
    for written, wanted in zip(back.layers, setup.layers, strict=True):
        assert written.raw == wanted.raw
    assert not (tmp_path / "run" / "JV.dat").exists()


def test_the_files_keep_their_layout(tmp_path):
    """SimSS reads its files by position: every line count and every untouched line kept."""
    setup = variant_setups(1)[0]
    write_simss_inputs(setup, tmp_path / "run")
    for name in ("simulation_setup.txt", "L1_parameters.txt", "L2_parameters.txt"):
        before = (setup.directory / name).read_text().splitlines()
        after = (tmp_path / "run" / name).read_text().splitlines()
        assert len(before) == len(after)
        changed = [a for a, b in zip(before, after, strict=True) if a != b]
        assert all("=" in line for line in changed)
