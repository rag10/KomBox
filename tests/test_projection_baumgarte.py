# tests/test_f24_features.py
import torch
import pytest

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver
from kombox.core.solvers_trapezoidal import TrapezoidalSolver
from kombox.blocks.mechanical import Mass1D


def test_projection_uses_externals_at_t1():
    """
    Verifica que la proyección pos-paso usa externals evaluados en t+dt.
    Creamos una restricción global: g(x) = x - F(t) = 0, con F(t) = 0 en t=0
    y F(t) = 1 en t>=dt. Tras el paso y la proyección, esperamos x ≈ 1.
    """
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B, dt = 2, 0.01
    dtype = torch.float32

    # Modelo: masa 1D con fuerza externa conectada
    m = Model("proj_ext_t1")
    m.add_block("mass", Mass1D(m=1.0))
    m.connect("F", "mass.F")

    # g(x) = x - F(t)
    def g_x_equals_F(t, states, inbuf, model, z):
        x = states["mass"][:, 0:1]
        F = inbuf["mass"]["F"]
        return x - F

    m.add_constraint_eq("x_equals_F", g_x_equals_F)
    m.build()

    # Estados iniciales: x=0, v=0
    x0 = torch.zeros((B, 1), dtype=dtype)
    v0 = torch.zeros((B, 1), dtype=dtype)
    m.initialize(batch_size=B, device=device, dtype=dtype, initial_states={"mass": {"x": x0, "v": v0}})

    # Externals: F(t) = 0 para t<dt, F(t) = 1 para t>=dt
    def ext_step(t, k):
        val = 0.0 if t < dt else 1.0
        return {"F": {"F": torch.full((B, 1), float(val), dtype=dtype)}}

    sim = Simulator(m, solver=RK45Solver())
    sim.enable_constraint_projection(True, tol=1e-9, max_iter=5, damping=1e-8)

    # Paso único: la proyección debe leer F en t+dt (1.0) y clavar x≈1
    sim.step(dt, externals_fn=ext_step)
    x = m.states["mass"][:, 0:1].detach()
    assert torch.allclose(x, torch.ones_like(x), atol=1e-5), f"x={x}"


def test_projection_cg_mode_forced():
    """
    Fuerza el modo CG (JVP/VJP) en la proyección estableciendo el umbral a 0.
    Restringimos g(x)=x=0 y comprobamos que clampa x→0 correctamente.
    """
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B, dt = 3, 0.05
    dtype = torch.float32

    m = Model("proj_cg")
    m.add_block("mass", Mass1D(m=1.0))
    m.connect("F", "mass.F")  # fuerza externa, pero la ponemos a 0
    def g_pos_zero(t, states, inbuf, model, z):
        return states["mass"][:, 0:1]  # g=0 => x=0
    m.add_constraint_eq("x_equals_0", g_pos_zero)
    m.build()

    # Estado inicial x != 0 para que se vea la corrección
    x0 = torch.tensor([[0.5], [0.3], [0.1]], dtype=dtype)
    v0 = torch.zeros((B, 1), dtype=dtype)
    m.initialize(batch_size=B, device=device, dtype=dtype, initial_states={"mass": {"x": x0, "v": v0}})

    def ext_zero(t, k):
        return {"F": {"F": torch.zeros((B, 1), dtype=dtype)}}

    sim = Simulator(m, solver=RK45Solver())
    sim.enable_constraint_projection(True, tol=1e-9, max_iter=5, damping=1e-8)
    # Forzar CG siempre
    sim._proj_cg_threshold = 0
    sim._proj_cg_maxit = 100
    sim._proj_cg_tol = 1e-12

    sim.step(dt, externals_fn=ext_zero)
    x = m.states["mass"][:, 0:1].detach()
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-6), f"x={x}"


def test_trapezoidal_baumgarte_smoke():
    """
    Smoke test: activar Baumgarte en Trapezoidal no debe romper el step.
    (Este test no verifica 'efecto dinámico' porque, sin z acoplado a fuerzas
    de restricción, el residual estabilizado no modifica estados.)
    """
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B, dt = 2, 0.02
    dtype = torch.float32

    m = Model("trap_baumgarte")
    m.add_block("mass", Mass1D(m=1.0))
    m.connect("F", "mass.F")
    def g_pos_zero(t, states, inbuf, model, z):
        return states["mass"][:, 0:1]  # x=0
    m.add_constraint_eq("x_equals_0", g_pos_zero)
    m.build()

    x0 = torch.tensor([[0.2], [0.1]], dtype=dtype)
    v0 = torch.zeros((B, 1), dtype=dtype)
    m.initialize(batch_size=B, device=device, dtype=dtype, initial_states={"mass": {"x": x0, "v": v0}})

    def ext_zero(t, k):
        return {"F": {"F": torch.zeros((B, 1), dtype=dtype)}}

    from kombox.core.algebraic.newton_krylov import NewtonKrylov
    alg = NewtonKrylov(mode="jfnk", max_iter=10, tol=1e-10)
    solver = TrapezoidalSolver(
        algebraic_solver=alg,
        baumgarte_enabled=True,
        baumgarte_alpha=2.0,
        baumgarte_beta=10.0,
    )

    sim = Simulator(m, solver=solver)
    # proyección off para que el test sea solo de "no rompe"
    sim.enable_constraint_projection(False)

    # Si no lanza excepción y devuelve outs coherentes, el smoke pasa
    outs = sim.step(dt, externals_fn=ext_zero)
    assert "mass" in outs and isinstance(outs["mass"], dict)
