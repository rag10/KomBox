# kombox/examples/example_008_sweeps_gpu.py
from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver  # también funciona en GPU
from kombox.core.utils import sweep_lin, sweep_log, sweep_list, combine_grids, apply_parameter_grid
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder
from kombox.utils.visualization import print_model_overview, visualize_model

import matplotlib.pyplot as plt

def build_msd() -> Model:
    m = Model("msd_sweeps_gpu")
    m.add_block("spring", Spring(k=40.0, x0=0.0))
    m.add_block("damper", Damper(c=1.0))
    m.add_block("sum",    Adder(n_inputs=3, width=1))
    m.add_block("mass",   Mass1D(m=1.0))

    m.blocks["sum"].alias_inputs({"Fs":"in1","Fd":"in2","Fext":"in3"}).alias_outputs({"F":"out"})
    m.connect("spring.F", "sum.Fs")
    m.connect("damper.F", "sum.Fd")
    m.connect("sum.F",    "mass.F")
    m.connect("mass.x",   "spring.x")
    m.connect("mass.v",   "damper.v")
    m.connect("Fext",     "sum.Fext")
    m.build()
    return m


def make_step_force_fn(B: int, t_step: float, amp: float, device: torch.device):
    # Devuelve tensores directamente en 'device'
    def fn(t: float, k: int):
        val = 0.0 if t < t_step else amp
        return {"Fext": {"Fext": torch.full((B,1), float(val), dtype=torch.float32, device=device)}}
    return fn


def main():
    torch.set_default_dtype(torch.float32)

    # --- Selección de dispositivo ---
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[KomBox] Usando dispositivo: {device}")

    model = build_msd()
    
    print_model_overview(model)
    fig, ax = visualize_model(model, include_io=True, layout="spring")
    plt.show()

    # --- Barridos (grids) ---
    g_k  = sweep_lin("spring.k", 10.0, 100.0, 6)      # 6 valores lineales
    g_c  = sweep_log("damper.c", -2, 1, 4)            # 1e-2 .. 1e1
    g_m  = sweep_list("mass.m",  [0.5, 1.0, 2.0])     # lista arbitraria

    grid = combine_grids(g_k, g_c, g_m)               # orden = k (lento), c, m (rápido)
    B = apply_parameter_grid(model, grid, device=device, dtype=torch.float32)
    print("Batch total =", B)  # 6*4*3 = 72

    # --- C.I. en GPU/CPU según 'device' ---
    x0 = torch.zeros((B,1), dtype=torch.float32, device=device)
    v0 = torch.zeros((B,1), dtype=torch.float32, device=device)
    model.initialize(batch_size=B, device=device, dtype=torch.float32,
                     initial_states={"mass": {"x": x0, "v": v0}})

    # --- Solver en GPU (funciona con RK45 y RK4/Euler). También puedes usar TorchDiffEqSolver. ---
    sim = Simulator(model, solver=RK45Solver())

    # --- Externos (escalón) en 'device' ---
    ext_fn = make_step_force_fn(B, t_step=0.010, amp=1.0, device=device)

    # --- Simulación ---
    dt = 2e-4
    T  = 0.5
    sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, reset_time=True)

    # --- Resultados: mover a CPU para imprimir/guardar ---
    x = sim.states["mass"][:, 0].detach().to("cpu")  # (B,)
    print("Primeras 10 |x(T)|:", x.abs()[:10].tolist())


if __name__ == "__main__":
    main()
