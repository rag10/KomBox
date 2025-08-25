from __future__ import annotations
import torch
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder

def build_msd() -> Model:
    m = Model("msd")
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
    m.connect("Fext",     "sum.Fext")  # external como nombre simple
    m.build()
    return m

def make_step_force_fn(B: int, t_step: float, amp: float):
    def fn(t: float, k: int):
        val = 0.0 if t < t_step else amp
        return {"Fext": {"Fext": torch.full((B,1), float(val), dtype=torch.float32)}}
    return fn

def test_checkpoint_preserves_time_and_progress():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    B = 2
    dt = 5e-4
    steps_total = 4000
    steps_half = steps_total // 2
    ext_fn = make_step_force_fn(B, t_step=0.010, amp=1.0)

    x0 = torch.tensor([[0.0],[0.02]], dtype=torch.float32)
    v0 = torch.zeros((B,1), dtype=torch.float32)
    init_states = {"mass": {"x": x0, "v": v0}}

    # Referencia
    m_ref = build_msd(); m_ref.initialize(batch_size=B, device=device, dtype=torch.float32, initial_states=init_states)
    sim_ref = Simulator(m_ref)
    sim_ref.simulate(dt=dt, steps=steps_total, externals_fn=ext_fn)  # reset_time=True por defecto
    t_ref, k_ref = sim_ref.t, sim_ref.k
    states_ref = {k: v.clone() for k, v in m_ref.states.items()}

    # Mitad → checkpoint → resto (OJO reset_time=False en la segunda simulación)
    m = build_msd(); m.initialize(batch_size=B, device=device, dtype=torch.float32, initial_states=init_states)
    sim = Simulator(m)
    sim.simulate(dt=dt, steps=steps_half, externals_fn=ext_fn)           # reset_time=True (por defecto)
    chk = sim.make_checkpoint(detach=False)

    assert abs(chk["t"] - steps_half*dt) < 1e-12
    assert chk["k"] == steps_half

    sim.restore_checkpoint(chk)
    assert abs(sim.t - steps_half*dt) < 1e-12
    assert sim.k == steps_half

    sim.simulate(dt=dt, steps=steps_total - steps_half, externals_fn=ext_fn, reset_time=False)

    assert abs(sim.t - t_ref) < 1e-12
    assert sim.k == k_ref
    for name in states_ref:
        assert torch.allclose(m.states[name], states_ref[name], rtol=1e-6, atol=1e-6), name
