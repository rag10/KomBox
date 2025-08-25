from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import TorchDiffEqSolver
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder


def build_msd() -> Model:
    m = Model("msd_torchdiffeq")
    m.add_block("spring", Spring(k=40.0, x0=0.0))
    m.add_block("damper", Damper(c=1.2))
    m.add_block("sum",    Adder(n_inputs=3, width=1))
    m.add_block("mass",   Mass1D(m=1.0))

    # Alias legibles
    m.blocks["sum"].alias_inputs({"Fs":"in1","Fd":"in2","Fext":"in3"}).alias_outputs({"F":"out"})

    # Conexiones internas
    m.connect("spring.F", "sum.Fs")
    m.connect("damper.F", "sum.Fd")
    m.connect("sum.F",    "mass.F")
    m.connect("mass.x",   "spring.x")
    m.connect("mass.v",   "damper.v")

    # Entrada externa (fuerza)
    m.connect("Fext",     "sum.Fext")
    # (opcional) salida externa del modelo
    m.connect("mass.x",   "x")

    m.build()
    return m


def make_step_force_fn(B: int, t_step: float, amp: float):
    # Para compatibilidad, acepta (t,k) pero solo usa t
    def fn(t: float, k: int):
        val = 0.0 if t < t_step else amp
        return {"Fext": {"Fext": torch.full((B,1), float(val), dtype=torch.float32)}}
    return fn


def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    model = build_msd()

    # batch de 3 condiciones iniciales
    B = 3
    x0 = torch.tensor([[0.0],[0.02],[-0.01]], dtype=torch.float32)
    v0 = torch.zeros((B,1), dtype=torch.float32)
    model.initialize(batch_size=B, device=device, dtype=torch.float32,
                     initial_states={"mass":{"x":x0, "v":v0}})

    # Solver global torchdiffeq (Dormand–Prince dopri5)
    solver = TorchDiffEqSolver(method="dopri5", use_adjoint=False, rtol=1e-5, atol=1e-7)
    sim = Simulator(model, solver=solver)

    # Fuerza escalón a 10 ms
    ext_fn = make_step_force_fn(B, t_step=0.010, amp=1.0)

    # Simular (el “dt externo” es el intervalo de muestreo; dentro torchdiffeq usa subpasos adaptativos)
    dt = 1e-3
    T  = 2.0
    sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, reset_time=True)

    # Estados finales
    x = sim.states["mass"][:, 0:1]
    v = sim.states["mass"][:, 1:2]
    print(f"[torchdiffeq] t_final = {sim.t:.6f} s, k = {sim.k}")
    print("x(B) =", x.squeeze(-1).tolist())
    print("v(B) =", v.squeeze(-1).tolist())


if __name__ == "__main__":
    main()
