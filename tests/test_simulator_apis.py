import torch
from kombox.core.model import Model
from kombox.core.block import ContinuousBlock
from kombox.core.simulator import Simulator

class ConstVel(ContinuousBlock):
    def __init__(self, v0=1.0):
        super().__init__()
        self.declare_io(inputs={}, outputs={"y": 1})
        self.v0 = float(v0)
    def state_size(self): return 1
    def init_state(self, B, device=None, dtype=None): return torch.zeros((B,1), device=device, dtype=dtype)
    def ode(self, state, inputs, t): return torch.full_like(state, self.v0), {"y": state}

def test_simulate_total_time_vs_steps_equivalence():
    B, dt = 1, 1e-3
    m1 = Model("a"); m1.add_block("X", ConstVel(2.0)); m1.build(); m1.initialize(batch_size=B, device=torch.device("cpu"), dtype=torch.float32)
    m2 = Model("b"); m2.add_block("X", ConstVel(2.0)); m2.build(); m2.initialize(batch_size=B, device=torch.device("cpu"), dtype=torch.float32)
    sim1 = Simulator(m1); sim1.simulate(dt=dt, total_time=0.1)          # reset_time=True
    sim2 = Simulator(m2); sim2.simulate(dt=dt, steps=int(0.1/dt))       # idem
    assert abs(sim1.t - sim2.t) < 1e-12 and sim1.k == sim2.k
    assert torch.allclose(m1.states["X"], m2.states["X"], rtol=1e-7, atol=1e-8)
