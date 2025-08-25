# examples/example_001_msd.py
from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder
from kombox.core.recorders import MemoryRecorder
from kombox.utils.externals import make_external_step, combine_externals

import matplotlib.pyplot as plt

def build_msd() -> Model:
    m = Model("msd")

    # bloques
    m.add_block("spring", Spring(k=40.0, x0=0.0))
    m.add_block("damper", Damper(c=1.2))
    m.add_block("sum",    Adder(n_inputs=3, width=1))
    m.add_block("mass",   Mass1D(m=1.0))

    # alias del sumador (legibles)
    m.blocks["sum"].alias_inputs({"Fs": "in1", "Fd": "in2", "Fext": "in3"}).alias_outputs({"F": "out"})

    # wiring interno
    m.connect("spring.F", "sum.Fs")
    m.connect("damper.F", "sum.Fd")
    m.connect("sum.F",    "mass.F")
    m.connect("mass.x",   "spring.x")
    m.connect("mass.v",   "damper.v")

    # declarar I/O externos del modelo
    m.connect("Fext",     "sum.Fext")   # entrada externa
    m.connect("mass.x",   "x")          # salida externa (opcional)

    return m

def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    # 1) modelo
    model = build_msd()
    model.build()  # infiere feedthrough, valida lazos y fija orden

    # 2) inicialización (B=3)
    B = 3
    x0 = torch.tensor([[0.00], [0.02], [-0.01]], dtype=torch.float32, device=device)
    v0 = torch.zeros((B, 1), dtype=torch.float32, device=device)
    states0 = {"mass": {"x": x0, "v": v0}}
    model.initialize(batch_size=B, device=device, dtype=torch.float32, initial_states=states0)

    # 3) simulador (se pasa states ya creados)
    sim = Simulator(model, validate_io=True, strict_numerics=False)

    # 4) entrada externa: escalón en 10 ms
    Fext = make_external_step("Fext", port="Fext", t0=0.01, y_before=0.0, y_after=3.0,
                              batch_size=B, width=1, device=device, dtype=torch.float32)
    ext_fn = combine_externals(Fext)

    rec = MemoryRecorder(
        model=model,
        signals=[
            "mass.x", "mass.v",
            "spring.F", "damper.F",
        ],
        store_time=True,
        detach_to_cpu=True,
    )

    # 5) simulación
    dt = 1e-4
    T  = 5.0
    sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, recorder=rec, reset_time=True, progress=True)

    # 6) resultados finales
    x = sim.states["mass"][:, 0:1]
    v = sim.states["mass"][:, 1:2]
    outs = {"x": x.squeeze(-1).tolist(), "v": v.squeeze(-1).tolist()}

    print(f"t_final = {sim.t:.6f} s")
    print("x_final =", [round(float(xx), 6) for xx in x.squeeze(-1)])
    print("v_final =", [round(float(vv), 6) for vv in v.squeeze(-1)])


    # 7) plot records
    packed = rec.stacked()
    t           = packed["t"] 
    x           = packed["mass.x"] 
    v           = packed["mass.v"] 
    F_spring    = packed["spring.F"]
    F_damper    = packed["damper.F"] 

    fig, axs = plt.subplots(3, figsize=(10, 6), sharex=True)
    for i in range(B):
        axs[0].plot(t, x[:,i,0], label=f"x (batch {i})", color=f"C{i}")
        axs[1].plot(t, v[:,i,0], label=f"v (batch {i})", color=f"C{i}")
        axs[2].plot(t, F_spring[:,i,0], label=f"F_spring (batch {i})", color=f"C{i}", linestyle='--')
        axs[2].plot(t, F_damper[:,i,0], label=f"F_damper (batch {i})", color=f"C{i}", linestyle=':')
    axs[0].set_ylabel("Position x [m]")
    axs[1].set_ylabel("Velocity v [m/s]")
    axs[2].set_ylabel("Forces [N]")
    axs[2].set_xlabel("Time [s]")    
    for ax in axs:
        ax.legend(loc="best", fontsize=8)   
        ax.grid()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
