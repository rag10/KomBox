import torch
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers_trapezoidal import TrapezoidalSolver
from kombox.core.algebraic.newton_krylov import NewtonKrylov
from kombox.blocks.mechanical import Mass1D

def test_kkt_mass1d_x_equals_zero_trapezoidal():
    """
    KKT holonómico: g(x)=x=0. La fuerza de restricción entra por el puerto 'F' de Mass1D
    como Phi_q^T * lambda = (1)*lambda. Sin proyección, la integración implícita debe
    mantener x ~ 0 (dentro de tolerancias) tras un paso.
    """
    torch.set_default_dtype(torch.float32)
    B = 3
    dtype = torch.float32
    device = torch.device("cpu")

    m = Model("kkt_1d")
    m.add_block("mass", Mass1D(m=1.0))
    m.connect("F", "mass.F")

    # g(x) = x
    def g_pos_zero(t, states, inbuf, model, z):
        return states["mass"][:, 0:1]  # (B,1)

    m.add_constraint_eq("x_equals_0", g_pos_zero)

    # Hook de fuerza: Phi_q^T * lambda = 1 * lambda
    def hook_force(t, states, inbuf, model, z, lam_i):
        return {"mass": {"F": lam_i}}  # (B,1) suma sobre 'F'
    m.add_constraint_force("x_equals_0", hook_force)

    m.build()

    # x0 != 0 para ver la corrección dinámica vía lambda; v0 = 0
    x0 = torch.tensor([[0.2],[ -0.1],[0.05]], dtype=dtype)
    v0 = torch.zeros((B,1), dtype=dtype)
    m.initialize(batch_size=B, device=device, dtype=dtype, initial_states={"mass":{"x":x0,"v":v0}})

    # No fuerzas externas reales: F=0
    def ext_zero(t,k): return {"F":{"F": torch.zeros((B,1), dtype=dtype)}}

    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30)
    # KKT puro (g=0), sin Baumgarte:
    trap = TrapezoidalSolver(algebraic_solver=alg, baumgarte_enabled=False)
    
    sim = Simulator(m, solver=trap)
    sim.enable_constraint_projection(False)  # queremos ver KKT puro

    dt = 0.02
    sim.step(dt, externals_fn=ext_zero)

    x = m.states["mass"][:, 0:1].detach()
    # Debe estar muy cerca de 0 (la dinámica implícita con lambda lo clava)
    assert torch.allclose(x, torch.zeros_like(x), atol=5e-4), f"x={x}"
