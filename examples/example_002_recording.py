# examples/example_002_recording.py
from __future__ import annotations
import os, sys, torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.recorders import MemoryRecorder, NPZChunkRecorder
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder


def build_msd() -> Model:
    m = Model("msd")
    m.add_block("spring", Spring(k=40.0, x0=0.0))
    m.add_block("damper", Damper(c=1.2))
    m.add_block("sum",    Adder(n_inputs=3, width=1))
    m.add_block("mass",   Mass1D(m=1.0))

    m.blocks["sum"].alias_inputs({"Fs": "in1", "Fd": "in2", "Fext": "in3"}).alias_outputs({"F": "out"})

    m.connect("spring.F", "sum.Fs")
    m.connect("damper.F", "sum.Fd")
    m.connect("sum.F",    "mass.F")
    m.connect("mass.x",   "spring.x")
    m.connect("mass.v",   "damper.v")

    # I/O externos
    m.connect("Fext",     "sum.Fext")
    m.connect("mass.x",   "x")
    return m


def make_step_force_fn(B: int, t_step: float, amp: float):
    def fn(t: float, k: int):
        val = 0.0 if t < t_step else amp
        return {"Fext": {"Fext": torch.full((B, 1), float(val))}}
    return fn


def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    model = build_msd()
    model.build()

    B = 3
    x0 = torch.tensor([[0.0], [1.0], [-0.3]], device=device)
    v0 = torch.zeros((B, 1), device=device)
    states0 = {"mass": {"x": x0, "v": v0}}
    model.initialize(batch_size=B, device=device, dtype=torch.float32, initial_states=states0)

    sim = Simulator(model, validate_io=True, strict_numerics=False)

    # --- Recorder en memoria (para inspección rápida) ---
    memrec = MemoryRecorder(
        model,
        signals=["mass.x", "sum.F"],         # outs (nota: "sum.F" es alias de "sum.out")
        states={"mass": ["x", "v"]},         # partes del estado
        store_time=True,
    )

    # --- Recorder NPZ por chunks (streaming) ---
    npzrec = NPZChunkRecorder(
        model,
        path_pattern=os.path.join(os.path.dirname(__file__), "msd_trace_part_{part:04d}.npz"),
        chunk_size=2000,
        signals=["mass.x", "sum.F"],
        states={"mass": ["x", "v"]},
        store_time=True,
        compress=True,
    )

    # Externos: escalón
    ext_fn = make_step_force_fn(B, t_step=0.010, amp=1.0)

    # Simulación
    dt = 1e-4
    T = 2.0
    sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, recorder=memrec, reset_time=True)
    # segunda pasada grabando a NPZ (solo para demostrar el uso; normalmente usarías uno u otro)
    sim.reset_time()
    sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, recorder=npzrec, reset_time=False)

    # Post: obtener arrays (T,B,n) del recorder en memoria
    data = memrec.stacked()
    print("Claves registradas:", list(data.keys()))
    print("t.shape =", data["t"].shape)
    print("mass.x shape =", data["mass.x"].shape)   # (T,B,1)
    print("sum.F shape  =", data["sum.F"].shape)    # (T,B,1)

if __name__ == "__main__":
    main()
