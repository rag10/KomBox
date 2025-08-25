# kombox/examples/example_012_identification_kc.py
from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from typing import List, Dict

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import TorchDiffEqSolver
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder


def build_msd() -> Model:
    """
    MSD:
      Fs = -k (x - x0)
      Fd = -c v
      F  = Fs + Fd + Fext
      Mass: dx=v ; dv=F/m ; outs x,v
    """
    m = Model("msd_id_kc")
    m.add_block("spring", Spring(k=40.0, x0=0.0))
    m.add_block("damper", Damper(c=1.2))
    m.add_block("sum",    Adder(n_inputs=3, width=1)).alias_inputs({"Fs":"in1","Fd":"in2","Fext":"in3"}).alias_outputs({"F":"out"})
    m.add_block("mass",   Mass1D(m=1.0))
    # wiring
    m.connect("spring.F", "sum.Fs")
    m.connect("damper.F", "sum.Fd")
    m.connect("sum.F",    "mass.F")
    m.connect("mass.x",   "spring.x")
    m.connect("mass.v",   "damper.v")
    # externals
    m.connect("Fext",     "sum.Fext")
    m.connect("mass.x",   "x")
    m.build()
    return m


def make_step_force_fn(B: int, amps: torch.Tensor, t_step: float = 0.010):
    """
    Escalón con amplitud por muestra (B,1). Compatible con Simulator.externals_fn(t,k).
    """
    def fn(t: float, _k: int):
        on = (t >= t_step)
        F = amps if on else torch.zeros_like(amps)
        return {"Fext": {"Fext": F}}
    return fn


@torch.no_grad()
def simulate_final_x(model: Model, solver: TorchDiffEqSolver, B: int,
                     x0: torch.Tensor, v0: torch.Tensor, amps: torch.Tensor,
                     dt: float, T: float) -> torch.Tensor:
    """
    Simula y devuelve x(T) sin grafo (datos 'verdad').
    """
    model.initialize(batch_size=B, device=amps.device, dtype=amps.dtype,
                     initial_states={"mass": {"x": x0, "v": v0}})
    sim = Simulator(model, solver=solver)
    ext_fn = make_step_force_fn(B, amps)
    sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, reset_time=True)
    return model.states["mass"][:, 0:1].clone()  # (B,1)


def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")  # cambia a "cuda" si tienes GPU
    dtype  = torch.float32

    # ------------------ Datos (ground truth) ------------------
    B = 32
    amps_true = torch.linspace(0.5, 2.0, B, device=device, dtype=dtype).reshape(B,1)
    x0 = torch.zeros((B,1), device=device, dtype=dtype)
    v0 = torch.zeros((B,1), device=device, dtype=dtype)
    dt, T = 1e-3, 1.0

    k_true, c_true = 40.0, 1.2

    model_true = build_msd()
    model_true.blocks["spring"].set_param("k", k_true)
    model_true.blocks["damper"].set_param("c", c_true)
    solver_data = TorchDiffEqSolver(method="dopri5", use_adjoint=False, rtol=1e-7, atol=1e-9)
    xT_true = simulate_final_x(model_true, solver_data, B, x0, v0, amps_true, dt, T)  # (B,1)

    # ------------------ Modelo a ajustar (k, c) ------------------
    model = build_msd()
    model.blocks["spring"].set_param("k", 15.0)   # mal a propósito
    model.blocks["damper"].set_param("c", 0.2)    # mal a propósito

    model.blocks["spring"].make_param_trainable("k", True)
    model.blocks["damper"].make_param_trainable("c", True)

    # Nota: initialize expandirá (1,1) -> (B,1); entrenarás B copias (válido).
    model.initialize(batch_size=B, device=device, dtype=dtype,
                     initial_states={"mass": {"x": x0, "v": v0}})

    solver = TorchDiffEqSolver(method="dopri5", use_adjoint=True, rtol=1e-6, atol=1e-8)
    sim = Simulator(model, solver=solver)
    ext_fn = make_step_force_fn(B, amps_true)

    # Parámetros a optimizar
    k_param = model.blocks["spring"].param_k
    c_param = model.blocks["damper"].param_c
    opt = torch.optim.Adam([k_param, c_param], lr=8e-2)

    # Logs para trazar convergencia
    hist: Dict[str, List[float]] = {"loss": [], "k_mean": [], "k_std": [], "c_mean": [], "c_std": []}

    n_epochs = 120
    for epoch in range(1, n_epochs+1):
        # Re-inicializa estados (NO resetea valores de k,c — son Parameters vivos)
        model.initialize(batch_size=B, device=device, dtype=dtype,
                         initial_states={"mass": {"x": x0, "v": v0}})
        sim.reset_time()

        # Forward con gradientes
        sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, reset_time=False)

        # Loss: MSE en x(T)
        xT = model.states["mass"][:, 0:1]
        loss = torch.mean((xT - xT_true)**2)

        # Backward
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Logging
        k_val = model.blocks["spring"].param_k.detach()
        c_val = model.blocks["damper"].param_c.detach()
        hist["loss"].append(float(loss.item()))
        hist["k_mean"].append(float(k_val.mean()))
        hist["k_std"].append(float(k_val.std()))
        hist["c_mean"].append(float(c_val.mean()))
        hist["c_std"].append(float(c_val.std()))

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}/{n_epochs}] loss={hist['loss'][-1]:.6e}  "
                  f"k=({hist['k_mean'][-1]:.3f}±{hist['k_std'][-1]:.3f})  "
                  f"c=({hist['c_mean'][-1]:.3f}±{hist['c_std'][-1]:.3f})")

    # Resultados
    print("\n--- Resultados ---")
    print(f"k_true={k_true:.3f}  |  k_est=({hist['k_mean'][-1]:.3f}±{hist['k_std'][-1]:.3f})")
    print(f"c_true={c_true:.3f}  |  c_est=({hist['c_mean'][-1]:.3f}±{hist['c_std'][-1]:.3f})")

    # (Opcional) guardar curva de convergencia en CSV simple
    # import pandas as pd
    # df = pd.DataFrame(hist)
    # df.to_csv("id_kc_history.csv", index=False)

if __name__ == "__main__":
    main()
