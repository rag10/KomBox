# kombox/tests/test_f2_scc_residual.py
import torch
import pytest

from kombox.core.model import Model
from kombox.blocks.basic import PassThrough  # asumiendo existe
from kombox.core.block import Block

def test_scc_two_node_cycle():
    m = Model("scc")
    m.add_block("A", PassThrough())
    m.add_block("B", PassThrough())
    m.connect("A.out", "B.in")
    m.connect("B.out", "A.in")
    m.build()
    sccs = m.analyze_topology()
    # Esperamos un SCC que contenga A y B
    assert any(set(comp) == {"A","B"} for comp in map(set, sccs))

class SumEq(Block):
    """Bloque puramente algebraico: g = a + b - c = 0"""
    def algebraic_residual(self, t, state, inputs, params):
        a = inputs["a"]; b = inputs["b"]; c = inputs["c"]
        return {"g": a + b - c}  # (B,1)

def test_build_residual_local_and_global():
    torch.set_default_dtype(torch.float32)
    B = 3
    a = torch.ones((B,1))
    b = 2*torch.ones((B,1))
    c = 3*torch.ones((B,1))

    m = Model("resid")
    m.add_block("E", SumEq())
    m.build()

    # Residual local: a + b - c = 0  (construido como 1+2-3=0)
    inbuf = {"E": {"a": a, "b": b, "c": c}}
    r_local, det = m.build_residual(t=0.0, inbuf=inbuf)
    assert r_local.shape == (B, 1)
    assert torch.allclose(r_local, torch.zeros_like(r_local))

    # Añadimos una restricción global: h = mean(a) - 1 = 0
    def h(t, states, inbuf, model):
        aa = inbuf["E"]["a"]  # (B,1)
        return (aa.mean(dim=0, keepdim=True)) - 1.0  # (1,1) -> se expandirá en cat

    m.add_constraint_eq("mean_a_eq_1", h)
    r_all, det2 = m.build_residual(t=0.0, inbuf=inbuf)
    # Tendremos 1 columna local + 1 columna global
    assert r_all.shape[1] == 2
    # la parte local sigue siendo cero
    assert torch.allclose(det2["E"], torch.zeros((B,1)))
