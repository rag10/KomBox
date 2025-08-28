# tests/test_projection_poststep.py
import torch
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver
from kombox.blocks.mechanical import Mass1D  # o mechanical, según tu árbol

def test_projection_clamps_state_to_constraint():
    """
    Comprobamos que la proyección pos-paso corrige el estado para satisfacer
    una restricción global g(x)=0 tras cada step.
    """
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B = 2
    dtype = torch.float32

    # externals F=0 (cierre captura B)
    def make_ext_zero(B):
        def ext(t, k):
            return {"F": {"F": torch.zeros((B, 1), dtype=dtype)}}
        return ext

    # Modelo mínimo: una masa con fuerza externa conectada.
    def make_model():
        m = Model("proj_test")
        m.add_block("mass", Mass1D(m=1.0))
        m.connect("F", "mass.F")  # <--- conectar external estricto
        # Restricción global: g(x) = x = 0  (clamp de la posición)
        def g_pos_zero(t, states, inbuf, model, z):
            x = states["mass"][:, 0:1]
            return x  # g=0 => x=0
        m.add_constraint_eq("x_equals_0", g_pos_zero)
        m.build()
        return m

    # Estados iniciales: x != 0 para que haya corrección; v=0
    x0 = torch.tensor([[0.5], [0.2]], dtype=dtype)
    v0 = torch.zeros((B, 1), dtype=dtype)
    init_states = {"mass": {"x": x0, "v": v0}}

    ext_zero = make_ext_zero(B)

    # --- Simulación SIN proyección ---
    m1 = make_model()
    m1.initialize(batch_size=B, device=device, dtype=dtype, initial_states=init_states)
    sim1 = Simulator(m1, solver=RK45Solver())
    sim1.step(0.1, externals_fn=ext_zero)  # F=0
    x_no_proj = m1.states["mass"][:, 0:1].detach()

    # --- Simulación CON proyección ---
    m2 = make_model()
    m2.initialize(batch_size=B, device=device, dtype=dtype, initial_states=init_states)
    sim2 = Simulator(m2, solver=RK45Solver())
    sim2.enable_constraint_projection(True, tol=1e-10, max_iter=5, damping=1e-8)
    sim2.step(0.1, externals_fn=ext_zero)  # F=0
    x_proj = m2.states["mass"][:, 0:1].detach()

    # Sin proyección: x ~ x0 (no cero)
    assert torch.all(torch.abs(x_no_proj) > 1e-3), f"x_no_proj={x_no_proj}"

    # Con proyección: x ≈ 0 (se clampa por la restricción global)
    assert torch.allclose(x_proj, torch.zeros_like(x_proj), atol=1e-6), f"x_proj={x_proj}"
    r,_ = m2.build_residual(sim2.t, states=m2.states, inbuf=sim2._inbuf)
    assert torch.allclose(r.abs().max(), torch.tensor(0.0), atol=1e-6)