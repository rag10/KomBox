# tests/test_externals_and_errors.py

import pytest
import torch
from kombox.core.model import Model
from kombox.core.block import ContinuousBlock
from kombox.core.simulator import Simulator

class PassThrough(ContinuousBlock):
    def __init__(self):
        super().__init__()
        self.declare_io(inputs={"u": 1}, outputs={"y": 1})
    def state_size(self): return 0
    def init_state(self, B, device=None, dtype=None): return torch.zeros((B,0), device=device, dtype=dtype)
    def ode(self, state, inputs, t): return torch.zeros_like(state), {"y": inputs["u"]}

def test_external_injection_ok():
    m = Model("ext"); m.add_block("P", PassThrough()); m.connect("sig", "P.u")
    m.build(); m.initialize(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    sim = Simulator(m)
    def ext(t,k): return {"sig": {"u": torch.tensor([[1.0],[2.0]], dtype=torch.float32)}}
    outs = sim.step(0.01, externals_fn=ext)
    assert torch.allclose(outs["P"]["y"], torch.tensor([[1.0],[2.0]]))

def test_external_missing_key_raises():
    m = Model("ext_err"); m.add_block("P", PassThrough()); m.connect("sig", "P.u")
    m.build(); m.initialize(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    sim = Simulator(m)
    def wrong_ext(t,k): return {"sig": {"wrong_port": torch.ones((1,1))}}
    with pytest.raises(KeyError) as ei:
        sim.step(0.01, externals_fn=wrong_ext)
    msg = str(ei.value)
    assert "External 'sig'" in msg and "'u'" in msg and "wrong_port" in msg
