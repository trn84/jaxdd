"""Least squares from many starts at once (M7): scipy's `least_squares` from every start, their
evaluations solved together.

`least_squares_many(residuals, theta0, lo, hi)` minimises `½·|r(θ)|²` within `lo ≤ θ ≤ hi` from
each row of `theta0` (starts × parameters) by `scipy.optimize.least_squares` (its trust-region
reflective method by default). `residuals` maps all starts at once, starts × parameters to
starts × residuals, each start's residuals depending on its own row alone (as
`validation/fit_measured.problem`'s `residuals_many`, every start's curves in one `solve_batch`).

Each start's optimiser runs in its own thread; its every request for residuals or Jacobian
waits until each start still running has asked, and then all are evaluated in one batch, the
residuals with the Jacobian (forward mode, one tangent per parameter for all starts together,
`residuals_and_jacobian`). The optimiser asks for the Jacobian where it has just evaluated the
residuals, and gets it from that evaluation, so an iteration of all starts costs one batched
evaluation. Each start's path is the one scipy takes from it alone, to the round-off of a batch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["ManyResult", "least_squares_many", "residuals_and_jacobian"]


def residuals_and_jacobian(residuals: Callable) -> Callable:
    """`θ ↦ (r, J)` for a batched `residuals` (starts × parameters → starts × residuals), `J`
    starts × residuals × parameters: one forward-mode tangent per parameter, the same unit
    tangent for every start at once (a start's residuals depend on its own row alone)."""

    @jax.jit
    def both(theta):
        P = theta.shape[1]
        basis = jnp.eye(P, dtype=theta.dtype)[:, None, :] * jnp.ones_like(theta)  # P × S × P
        r, cols = jax.vmap(lambda t: jax.jvp(residuals, (theta,), (t,)), out_axes=(None, 0))(basis)
        return r, jnp.moveaxis(cols, 0, -1)

    return both


@dataclass
class ManyResult:
    results: list  # scipy's `OptimizeResult` per start
    theta: np.ndarray  # starts × parameters
    cost: np.ndarray  # ½·|r|² per start
    calls: int  # batched evaluations of residuals and Jacobian
    seconds: float

    @property
    def best(self) -> int:
        return int(np.argmin(self.cost))


class _Batcher:
    """Collects one request per running start and evaluates them as one batch (the rows of the
    starts not asking hold a copy, for a fixed shape); keeps each start's last evaluation."""

    def __init__(self, both: Callable, starts: int, progress: Callable | None = None):
        self.both, self.S, self.progress = both, starts, progress
        self.cond = threading.Condition()
        self.pending: dict[int, np.ndarray] = {}
        self.answers: dict[int, tuple] = {}
        self.last: dict[int, tuple] = {}
        self.running = starts
        self.calls = 0

    def get(self, i: int, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        last = self.last.get(i)
        if last is not None and np.array_equal(last[0], theta):
            return last[1], last[2]
        with self.cond:
            self.pending[i] = np.array(theta, float)
            self.cond.notify_all()
            while i not in self.answers:
                self.cond.wait()
            r, J = self.answers.pop(i)
        self.last[i] = (np.array(theta, float), r, J)
        return r, J

    def done(self) -> None:
        with self.cond:
            self.running -= 1
            self.cond.notify_all()

    def serve(self) -> None:
        while True:
            with self.cond:
                while self.running and len(self.pending) < self.running:
                    self.cond.wait()
                if not self.pending:
                    return
                batch, self.pending = self.pending, {}
            rows = np.repeat(next(iter(batch.values()))[None], self.S, axis=0)
            for i, theta in batch.items():
                rows[i] = theta
            r, J = (np.asarray(a) for a in self.both(jnp.asarray(rows)))
            self.calls += 1
            if self.progress is not None:
                self.progress(self.calls, len(batch))
            with self.cond:
                for i in batch:
                    self.answers[i] = (r[i], J[i])
                self.cond.notify_all()


def least_squares_many(residuals: Callable, theta0, lo, hi, progress: Callable | None = None,
                       **kwargs) -> ManyResult:  # fmt: skip
    """Each start of `theta0` (starts × parameters) by `scipy.optimize.least_squares` within the
    box `[lo, hi]`, `kwargs` passed to it; `residuals` batched (module docstring).
    `progress(calls, asking)` after each batched evaluation, `asking` the starts in it."""
    from scipy.optimize import least_squares

    t0 = time.perf_counter()
    theta0 = np.asarray(theta0, float)
    S = theta0.shape[0]
    batcher = _Batcher(residuals_and_jacobian(residuals), S, progress)
    results: list = [None] * S
    errors: list = []

    def run(i):
        try:
            results[i] = least_squares(lambda t: batcher.get(i, t)[0], theta0[i],
                                       jac=lambda t: batcher.get(i, t)[1], bounds=(lo, hi),
                                       **kwargs)  # fmt: skip
        except BaseException as e:  # re-raised in the caller's thread
            errors.append(e)
        finally:
            batcher.done()

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(S)]
    for t in threads:
        t.start()
    batcher.serve()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return ManyResult(results=results, theta=np.stack([r.x for r in results]),
                      cost=np.array([r.cost for r in results]), calls=batcher.calls,
                      seconds=time.perf_counter() - t0)  # fmt: skip
