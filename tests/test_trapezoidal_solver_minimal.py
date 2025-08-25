# kombox/tests/test_trapezoidal_solver_minimal.py
import torch
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.blocks.basic import PassThrough

# Importa desde donde re-exportes TrapezoidalSolver y NewtonKrylov
from kombox.core.solvers_trapezoidal import TrapezoidalSolver
from kombox.core.algebraic.newton_krylov import NewtonKrylov


def test_trapezoidal_runs_and_is_global():
    m = Model("trap_min")
    m.add_block("P", PassThrough(width=1))
    m.build()
    m.initialize(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    alg = NewtonKrylov()
    solver = TrapezoidalSolver(algebraic_solver=alg)

    assert getattr(solver, "is_global", True), "Trapezoidal debe ser solver global"

    sim = Simulator(m, solver=solver)
    sim.step(0.1)
    sim.step(0.2)
