# The jaxdd handbook

How jaxdd works and how to use it. What is planned is in [PLAN.md](../PLAN.md); this handbook
describes what exists.

| Page | For |
|---|---|
| [Overview](overview.md) | what jaxdd is, why it batches, what it solves and what it refuses, the design decisions, the repository, licence and attribution |
| [Tutorial](tutorial.md) | installing; one device from SIMsalabim's files; many devices in a batch; reading the result; known limits |
| [The model](model.md) | the device equations and their discretisation: grid, Poisson, Scharfetter–Gummel, interfaces, recombination, ions, the current |
| [The solver](solver.md) | how a J-V curve is solved: the sequence of solves, the Newton step, the bordered system, the block sweep, convergence; flowcharts and pseudocode |
| [Batching](batching.md) | the same solver over many devices: pytrees, `vmap`, loops without branches, the lockstep, compilation |
| [Hardware](hardware.md) | CPU, NVIDIA GPUs, AMD MI50: how to run fast on each, and how to benchmark a machine |
| [Validation](validation.md) | the reference, the criteria, what the comparisons with SimSS show, the tests and scripts |
| [Differences from SIMsalabim](differences.md) | where jaxdd's answer differs from SimSS's on purpose, why, and the switch that reproduces SimSS |

A first reading: the overview, then the tutorial; the model and the solver when the numbers
need explaining.
