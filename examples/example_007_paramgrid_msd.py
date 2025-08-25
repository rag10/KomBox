from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver  # o TorchDiffEqSolver si lo tienes instalado
from kombox.core.utils import apply_parameter_grid
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder


def build_msd() -> Model:
    m = Model("msd_paramgrid")
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
    m.connect("Fext",     "sum.Fext")  # entrada externa
    m.build()
    return m


def make_step_force_fn(B: int, t_step: float, amp: float):
    def fn(t: float, k: int):
        val = 0.0 if t < t_step else amp
        return {"Fext": {"Fext": torch.full((B,1), float(val), dtype=torch.float32)}}
    return fn


def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    model = build_msd()

    # --- Definir la malla de parámetros (k, c, m) ---
    grid = {
        "spring.k": [20.0, 40.0, 60.0, 80.0],
        "damper.c": [0.2, 0.5, 1.0, 2.0],
        "mass.m":   [0.5, 1.0, 2.0, 4.0],
    }

    # Aplica el grid a los parámetros → crea batch (B = 4*4*4 = 64)
    B = apply_parameter_grid(model, grid, device=device, dtype=torch.float32)
    print("Batch total B =", B)

    # Condiciones iniciales (todas iguales, pero puedes variarlas por B si quieres)
    x0 = torch.zeros((B,1), dtype=torch.float32, device=device)
    v0 = torch.zeros((B,1), dtype=torch.float32, device=device)

    model.initialize(batch_size=B, device=device, dtype=torch.float32,
                     initial_states={"mass":{"x": x0, "v": v0}})

    # Solver y simulación
    sim = Simulator(model, solver=RK45Solver())
    ext_fn = make_step_force_fn(B, t_step=0.010, amp=1.0)

    dt = 1e-4
    T  = 0.5
    sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, reset_time=True)

    # Lee estados finales y muestra algunas filas (primeras 5 combinaciones)
    x = sim.states["mass"][:, 0:1]  # (B,1)
    v = sim.states["mass"][:, 1:2]
    print("x[:5] =", x[:5, 0].tolist())
    print("v[:5] =", v[:5, 0].tolist())

    # Ejemplo: calcular costo por batch (p.ej., |x(T)|) — listo para optimización
    cost = x.abs().squeeze(1)  # (B,)
    print("cost[:5] =", cost[:5].tolist())


if __name__ == "__main__":
    main()
