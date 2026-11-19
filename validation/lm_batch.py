"""scipy's `least_squares` from many starts at once, its evaluations gathered into batches.

jaxdd's transients cost a batch little more than one device: the time steps are sequential, and a
batch of a few hundred devices fills the GPU (docs/batching.md). A fit from several starts one
after another pays that floor for every start and every evaluation. Here each start runs its own
`least_squares` (the trust-region reflective method, its handling of bounds) in a thread of its
own; every residual and Jacobian it asks for goes to one `Gather`, which waits until every start
still running has asked and then solves all their parameter sets in one batch. Each start sees
the values it would have seen alone, so it takes the same path as one by one, to round-off.

    from lm_batch import least_squares_many
    fits, batches = least_squares_many(problem.residuals_many, starts, lo, hi)

`residuals_many(thetas) -> (R, ok)`: residuals (sets × points) and whether each set solved; a
set that did not is expected to carry a penalty in its residuals. The Jacobian is the forward
difference, a column whose step does not solve the backward one, else 0.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.optimize import least_squares


class Gather:
    """Requests of parameter sets from several threads, solved together once every thread still
    running has one waiting (`ask`), or a thread leaves (`leave`)."""

    def __init__(self, residuals_many, threads):
        self.residuals_many = residuals_many
        self.running = threads
        self.waiting = {}  # thread -> its sets
        self.answers = {}  # thread -> (R, ok)
        self.cond = threading.Condition()
        self.batches = []  # (sets, seconds) per batch

    def _solve(self):
        """Every waiting request in one batch; called with the lock held, by the last to ask."""
        order = list(self.waiting)
        sets = [x for k in order for x in self.waiting[k]]
        t0 = time.time()
        R, ok = self.residuals_many(sets)
        self.batches.append((len(sets), time.time() - t0))
        i = 0
        for k in order:
            n = len(self.waiting[k])
            self.answers[k] = (R[i : i + n], ok[i : i + n])
            i += n
        self.waiting.clear()
        self.cond.notify_all()

    def ask(self, who, sets):
        with self.cond:
            self.waiting[who] = list(sets)
            if len(self.waiting) == self.running:
                self._solve()
            while who not in self.answers:
                self.cond.wait()
            return self.answers.pop(who)

    def leave(self):
        with self.cond:
            self.running -= 1
            if self.waiting and len(self.waiting) == self.running:
                self._solve()


def least_squares_many(residuals_many, starts, lo, hi, h=0.02, max_nfev=60, log=print):
    """`least_squares` from every row of `starts` at once (`Gather`): a list of dicts per start
    (`start`, `theta`, `rms`, `nfev`, `status`, `message`, `seconds`) and the batches' sizes and
    wall times."""
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    gather = Gather(residuals_many, len(starts))
    fits = [None] * len(starts)
    E = np.eye(starts.shape[1])

    def run(k):
        t0 = time.time()

        def fun(x):
            return gather.ask(k, [x])[0][0]

        def jac(x):
            R, ok = gather.ask(k, [x] + [x + h * e for e in E])
            J = (R[1:] - R[0]) / h
            failed = np.flatnonzero(~ok[1:])
            if failed.size:  # the backward difference, else nothing in that direction
                Rb, okb = gather.ask(k, [x - h * E[i] for i in failed])
                for m, i in enumerate(failed):
                    J[i] = (R[0] - Rb[m]) / h if okb[m] else 0.0
            return J.T

        try:
            x0 = np.clip(starts[k], lo, hi)
            res = least_squares(fun, x0, jac=jac, bounds=(lo, hi), max_nfev=max_nfev)
            fits[k] = {"start": x0.tolist(), "theta": res.x.tolist(),
                       "rms": float(np.sqrt(np.mean(res.fun**2))), "nfev": int(res.nfev),
                       "status": int(res.status), "message": res.message,
                       "seconds": time.time() - t0}  # fmt: skip
            if log:
                log(f"start {k}: rms {fits[k]['rms']:.4f} after {res.nfev} evaluations in "
                    f"{fits[k]['seconds']:.0f} s ({res.message})")  # fmt: skip
        finally:
            gather.leave()

    threads = [threading.Thread(target=run, args=(k,)) for k in range(len(starts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return fits, gather.batches
