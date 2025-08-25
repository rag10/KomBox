# tests/test_torchdiffeq_vs_rk45.py
from __future__ import annotations
import torch
import pytest

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver, TorchDiffEqSolver
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder


@pytest.mark.skipif("torchdiffeq" not in [m.split('.')[0] for m in list(globals().get('__builtins__', {}))] and __import__('importlib').util.find_spec("torchdiffeq") is None, reason="torchdiffeq no instalado")
def test_torchdiffeq_close_to_rk45():
    def build_msd() -> Model:
        m = Model("msd_test")
        m.add_block("spring", Spring(k=40.0, x0=0.0))
        m.add_block("damper", Damper(c=1.2))
        m.add_block("sum",    Adder(n_inputs=3, width=1))
        m.add_block("mass",   Mass1D(m=1.0))
        m.blocks["sum"].alias_inputs({"Fs":"in1","Fd":"in2","Fext":"in3"}).alias_outputs({"F":"out"})
        m.connect("spring.F", "sum.Fs")
        m.connect("damper.F", "sum.Fd")
        m.connect("sum.F",    "mass.F")
        m.connect("mass.x",   "spring.x")
        m.connect("mass.v",   "damper.v")
        m.connect("Fext",     "sum.Fext")
        m.build()
        return m

    def make_step_force_fn(B: int, t_step: float, amp: float):
        def fn(t: float, k: int):
            val = 0.0 if t < t_step else amp
            return {"Fext": {"Fext": torch.full((B,1), float(val), dtype=torch.float32)}}
        return fn

    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    B = 2
    x0 = torch.tensor([[0.0],[0.02]], dtype=torch.float32)
    v0 = torch.zeros((B,1), dtype=torch.float32)
    init_states = {"mass": {"x": x0, "v": v0}}

    ext_fn = make_step_force_fn(B, t_step=0.010, amp=1.0)

    dt_ref = 2e-4
    T = 0.5
    steps = int(T/dt_ref)

    # Referencia RK45 (por bloque)
    m_ref = build_msd(); m_ref.initialize(B, device=device, dtype=torch.float32, initial_states=init_states)
    sim_ref = Simulator(m_ref, solver=RK45Solver()); sim_ref.simulate(dt=dt_ref, steps=steps, externals_fn=ext_fn)
    x_ref = m_ref.states["mass"][:, 0:1]; v_ref = m_ref.states["mass"][:, 1:2]

    # TorchDiffEq (global) con muestreo más grueso
    dt_tdq = 1e-3; steps_tdq = int(T/dt_tdq)
    m_tdq = build_msd(); m_tdq.initialize(B, device=device, dtype=torch.float32, initial_states=init_states)
    solver = TorchDiffEqSolver(method="dopri5", use_adjoint=False, rtol=1e-6, atol=1e-8)
    sim_tdq = Simulator(m_tdq, solver=solver); sim_tdq.simulate(dt=dt_tdq, steps=steps_tdq, externals_fn=ext_fn)
    x_tdq = m_tdq.states["mass"][:, 0:1]; v_tdq = m_tdq.states["mass"][:, 1:2]

    assert torch.allclose(x_tdq, x_ref, rtol=5e-3, atol=5e-3)
    assert torch.allclose(v_tdq, v_ref, rtol=5e-3, atol=5e-3)
