import torch
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers_trapezoidal import TrapezoidalSolver
from kombox.core.algebraic.newton_krylov import NewtonKrylov
from kombox.blocks.mechanical import Mass1D

def _build_mass1d_with_constraint(B, dtype):
    m = Model("kkt_mass1d")
    m.add_block("mass", Mass1D(m=1.0))
    m.connect("F", "mass.F")
    def g_pos_zero(t, states, inbuf, model, z):  # g=x
        return states["mass"][:, 0:1]
    m.add_constraint_eq("x_equals_0", g_pos_zero)
    def hook_force(t, states, inbuf, model, z, lam_i):
        return {"mass": {"F": lam_i}}
    m.add_constraint_force("x_equals_0", hook_force)
    m.build()
    x0 = torch.tensor([[0.2],[ -0.1],[0.05]], dtype=dtype)
    v0 = torch.zeros((B,1), dtype=dtype)
    m.initialize(batch_size=B, device=torch.device("cpu"), dtype=dtype, initial_states={"mass":{"x":x0,"v":v0}})
    def ext_zero(t,k): return {"F":{"F": torch.zeros((B,1), dtype=dtype)}}
    return m, ext_zero

def test_kkt_pure_clamps_x_to_zero():
    torch.set_default_dtype(torch.float32)
    B, dt = 3, 0.02
    dtype = torch.float32
    m, ext_zero = _build_mass1d_with_constraint(B, dtype)

    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30)
    trap = TrapezoidalSolver(algebraic_solver=alg, constraint_mode="kkt")
    sim = Simulator(m, solver=trap)
    sim.enable_constraint_projection(False)

    sim.step(dt, externals_fn=ext_zero)
    x = m.states["mass"][:, 0:1].detach()
    assert torch.allclose(x, torch.zeros_like(x), atol=5e-4), f"x={x}"

def test_kkt_baumgarte_reduces_error_not_exact():
    torch.set_default_dtype(torch.float32)
    B, dt = 3, 0.02
    dtype = torch.float32
    m, ext_zero = _build_mass1d_with_constraint(B, dtype)

    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30)
    trap = TrapezoidalSolver(algebraic_solver=alg, constraint_mode="kkt_baumgarte",
                             baumgarte_alpha=1.0, baumgarte_beta=10.0)
    sim = Simulator(m, solver=trap)
    sim.enable_constraint_projection(False)

    x_before = m.states["mass"][:, 0:1].detach().abs().max()
    sim.step(dt, externals_fn=ext_zero)
    x_after = m.states["mass"][:, 0:1].detach().abs().max()

    assert x_after < x_before, f"x_before={x_before}, x_after={x_after}"
    # no exigimos x≈0 exacto aquí

def test_kkt_baumgarte_with_projection_clamps_to_zero():
    torch.set_default_dtype(torch.float32)
    B, dt = 3, 0.02
    dtype = torch.float32
    m, ext_zero = _build_mass1d_with_constraint(B, dtype)

    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30)
    trap = TrapezoidalSolver(algebraic_solver=alg, constraint_mode="kkt_baumgarte",
                             baumgarte_alpha=1.0, baumgarte_beta=10.0)
    sim = Simulator(m, solver=trap)
    # proyección cada paso
    sim.enable_constraint_projection(True, every_n_steps=1, tol=1e-9, max_iter=5, damping=1e-8)

    sim.step(dt, externals_fn=ext_zero)
    x = m.states["mass"][:, 0:1].detach()
    assert torch.allclose(x, torch.zeros_like(x), atol=5e-4), f"x={x}"
