import pytest
import torch

nk = pytest.importorskip("kombox.core.algebraic.newton_krylov", reason="newton_krylov aún no disponible")
trap = pytest.importorskip("kombox.core.solvers_trapezoidal", reason="trapezoidal aún no disponible")

from kombox.core.model import Model
from kombox.core.block import ContinuousBlock
from kombox.core.simulator import Simulator

NewtonKrylov = nk.NewtonKrylov
TrapezoidalSolver = trap.TrapezoidalSolver


class LinearRelax(ContinuousBlock):
    """dx/dt = -a * x, y = x; con 'a' entrenable usando API de parámetros."""
    def __init__(self, a=0.5):
        super().__init__()
        self.declare_io(inputs={}, outputs={"y": 1})
        self.declare_param("a", a, trainable=True)

    def state_size(self): return 1

    def init_state(self, B, device=None, dtype=None):
        return torch.ones((B, 1), device=device, dtype=dtype)  # x0 = 1

    def ode(self, state, inputs, t):
        a = self.get_param("a")  # tensor trainable
        dx = -a * state
        return dx, {"y": state}


def test_newton_krylov_simple():
    alg = NewtonKrylov(tol=1e-10, max_iter=20)
    target = torch.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
    F = lambda z: z - target
    z0 = torch.zeros_like(target)
    z_star = alg.solve(F, z0)
    assert torch.allclose(z_star, target, atol=1e-6)


def test_trapezoidal_smoke_and_grad():
    m = Model(); m.add_block("P", LinearRelax(a=0.7)); m.build()
    m.initialize(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    alg = NewtonKrylov()
    solver = TrapezoidalSolver(algebraic_solver=alg)
    sim = Simulator(m, solver=solver)

    sim.step(0.1); sim.step(0.1)  # humo
    y = m.blocks["P"]._expose_outputs(m.states["P"], {}, sim.t)["y"]
    loss = (y**2).mean()
    loss.backward()

    a_param = m.blocks["P"].get_param("a")  # sin as_tensor
    assert getattr(a_param, "grad", None) is not None
    assert torch.isfinite(a_param.grad).all()
