# tests/test_dae_init.py
import torch
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.block import Block

# Bloque puramente algebraico: g = a + b - c = 0
class SumEq(Block):
    def __init__(self, width: int = 1):
        super().__init__()
        # Declaramos entradas canónicas: a, b, c
        self.declare_io(inputs={"a": width, "b": width, "c": width})
        # Sin salidas, sin estado

    def algebraic_residual(self, t, state, inputs, params):
        a = inputs["a"]; b = inputs["b"]; c = inputs["c"]
        return {"g": a + b - c}  # (B,1)

def test_initialize_consistent_pure_algebraic_ok():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B = 4
    a = torch.ones((B,1))
    b = 2*torch.ones((B,1))
    c = 3*torch.ones((B,1))

    m = Model("init_ok")
    m.add_block("E", SumEq())
    # Conectamos externals simples "a","b","c" a los puertos del bloque
    m.connect("a", "E.a")
    m.connect("b", "E.b")
    m.connect("c", "E.c")
    m.build()
    # IMPORTANTE: aunque el bloque no tenga estado, inicializamos para fijar B/device/dtype
    m.initialize(batch_size=B, device=device, dtype=torch.float32)

    def ext_ok(t, k):
        return {"a":{"a": a}, "b":{"b": b}, "c":{"c": c}}

    sim = Simulator(m)
    info = sim.initialize_consistent(t0=0.0, externals_fn=ext_ok, tol=1e-9)

    assert "residual" in info and "residual_norm" in info
    assert info["residual"].shape == (B, 1)
    assert info["residual_norm"] < 1e-6
    assert info["z_dim"] == 0
    assert info["did_solve"] is False


def test_initialize_consistent_pure_algebraic_not_ok():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B = 3
    a = torch.ones((B,1))
    b = 2*torch.ones((B,1))
    c = 2.9*torch.ones((B,1))  # no cumple a+b-c=0

    m = Model("init_bad")
    m.add_block("E", SumEq())
    m.connect("a", "E.a")
    m.connect("b", "E.b")
    m.connect("c", "E.c")
    m.build()
    m.initialize(batch_size=B, device=device, dtype=torch.float32)

    def ext_bad(t, k):
        return {"a":{"a": a}, "b":{"b": b}, "c":{"c": c}}

    sim = Simulator(m)
    info = sim.initialize_consistent(t0=0.0, externals_fn=ext_bad, tol=1e-9)

    assert info["residual"].shape == (B, 1)
    assert info["residual_norm"] > 1e-6
    assert info["z_dim"] == 0
    assert info["did_solve"] is False
