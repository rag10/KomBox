# tests/test_dae_init_z.py
import torch
from kombox.core.model import Model
from kombox.core.simulator import Simulator

def test_initialize_consistent_solves_z_global_constraint():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B = 5
    a = torch.full((B,1), 1.0)
    b = torch.full((B,1), 2.0)
    c = torch.full((B,1), 3.0)

    m = Model("init_z")
    # Declaramos que E no es necesario; solo constraints globales que usen z
    # Conectamos externals para poder usar 'inbuf' en la constraint
    m.build()  # sin bloques; válido
    m.declare_algebraic_dim(1)  # z ∈ R^1 (por batch)

    # Constraint global: g(z,a,b,c) = z - (a + b - c)  ->  z = a+b-c (=0 aquí)
    def g(t, states, inbuf, model, z):
        # Para este test, inyectamos externals directamente desde initialize_consistent
        aa = inbuf.get("a", {}).get("a", None)
        bb = inbuf.get("b", {}).get("b", None)
        cc = inbuf.get("c", {}).get("c", None)
        assert aa is not None and bb is not None and cc is not None
        # z: (B,1)
        return z - (aa + bb - cc)

    m.add_constraint_eq("z_equals_a_plus_b_minus_c", g)

    def externals(t, k):
        return {"a":{"a": a}, "b":{"b": b}, "c":{"c": c}}

    sim = Simulator(m)
    info = sim.initialize_consistent(t0=0.0, externals_fn=externals, tol=1e-12)

    assert info["z_dim"] == 1
    assert info["did_solve"] is True
    z = info["z_star"]
    assert z.shape == (B, 1)
    # a+b-c = 1+2-3 = 0
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-6)
    # residual final ~ 0
    assert info["residual_norm"] < 1e-8
