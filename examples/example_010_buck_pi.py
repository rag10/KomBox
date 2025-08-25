from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver  # o TorchDiffEqSolver
from kombox.blocks.basic import Adder, Gain
from kombox.blocks.control import PID
from kombox.blocks.power import BuckAveraged

def build_model() -> Model:
    m = Model("buck_pi")

    m.add_block("plant", BuckAveraged(L=100e-6, C=100e-6, Rload=10.0))
    m.add_block("neg",   Gain(gain=-1.0))
    m.add_block("sum",   Adder(n_inputs=2, width=1)).alias_inputs({"r":"in1","y":"in2"}).alias_outputs({"e":"out"})
    # Duty 0..1
    m.add_block("pi",    PID(Kp=0.02, Ki=200.0, Kd=0.0, u_min=0.0, u_max=1.0))

    # y = Vout
    m.connect("plant.Vout", "neg.in")
    m.connect("neg.out",    "sum.y")
    m.connect("Vref",       "sum.r")     # referencia Vref externa
    m.connect("sum.e",      "pi.e")
    m.connect("pi.u",       "plant.D")
    m.connect("Vin",        "plant.Vin") # Vin externa (p.ej., 12V)
    # R externo opcional (si no, usa Rload del bloque)
    # m.connect("R",          "plant.R")

    m.build()
    return m

def make_externals(B: int, Vref: float = 5.0, Vin: float = 12.0):
    def fn(t: float, k: int):
        vref = Vref if t >= 0.005 else 0.0
        vin  = Vin
        return {
            "Vref": {"r": torch.full((B,1), vref)},
            "Vin":  {"Vin":  torch.full((B,1), vin)},
        }
    return fn

def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    m = build_model()
    B = 4
    IL0 = torch.zeros((B,1)); V0 = torch.zeros((B,1))
    m.initialize(batch_size=B, device=device, dtype=torch.float32,
                 initial_states={"plant": {"IL": IL0, "Vout": V0}})

    sim = Simulator(m, solver=RK45Solver())  # o TorchDiffEqSolver()
    ext = make_externals(B, Vref=5.0, Vin=12.0)

    dt, T = 2e-5, 0.01
    sim.simulate(dt=dt, total_time=T, externals_fn=ext, reset_time=True)

    vout = sim.states["plant"][:, 1:2]
    print(f"t_final={sim.t:.6f}s, k={sim.k}")
    print("Vout[:5] =", vout[:5,0].tolist())

if __name__ == "__main__":
    main()
