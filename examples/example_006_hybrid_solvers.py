from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import TorchDiffEqSolver
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder
from kombox.core.block import DiscreteBlock


class DiscretePD(DiscreteBlock):
    """
    Control PD discreto (ZOH implícito): u_k = -Kp * x_k - Kd * v_k
    - Estado: guarda u_k (opcional, aquí 1 canal) para ilustrar 'update'.
    - Salida: 'u' (anchura 1).
    """
    def __init__(self, Kp=50.0, Kd=5.0):
        super().__init__()
        self.declare_io(inputs={"x":1, "v":1}, outputs={"u":1}, state_size=1)
        self.declare_param("Kp", Kp)
        self.declare_param("Kd", Kd)
        self.state_alias = {"u": 0}

    def update(self, state, inputs, dt, t):
        x = inputs["x"]; v = inputs["v"]
        Kp = self.get_param("Kp"); Kd = self.get_param("Kd")
        u = -(Kp * x + Kd * v)
        new_state = u  # ZOH: guardar último valor (ilustrativo)
        return new_state, {"u": u}


def build_hybrid_msd() -> Model:
    m = Model("msd_hybrid_pd")

    # Bloques continuos
    m.add_block("spring", Spring(k=40.0, x0=0.0))
    m.add_block("damper", Damper(c=1.2))
    m.add_block("mass",   Mass1D(m=1.0))
    m.add_block("sum",    Adder(n_inputs=3, width=1)).alias_inputs({"Fs":"in1","Fd":"in2","u":"in3"}) \
                                                     .alias_outputs({"F":"out"})
    # Control discreto
    m.add_block("pd",     DiscretePD(Kp=50.0, Kd=6.0))

    # Wiring
    m.connect("spring.F", "sum.Fs")
    m.connect("damper.F", "sum.Fd")
    m.connect("pd.u",     "sum.u")
    m.connect("sum.F",    "mass.F")
    m.connect("mass.x",   "spring.x")
    m.connect("mass.v",   "damper.v")
    m.connect("mass.x",   "pd.x")
    m.connect("mass.v",   "pd.v")

    # salida externa opcional
    m.connect("mass.x",   "x")

    m.build()
    return m


def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    model = build_hybrid_msd()

    # batch de 2 condiciones iniciales
    B = 2
    x0 = torch.tensor([[0.05],[ -0.03]], dtype=torch.float32)
    v0 = torch.zeros((B,1), dtype=torch.float32)
    model.initialize(batch_size=B, device=device, dtype=torch.float32,
                     initial_states={"mass":{"x":x0, "v":v0},
                                     "pd":{"u": torch.zeros((B,1))}})

    # Solver global (torchdiffeq); integrará la parte continua y aplicará PD discreto al final de cada dt.
    solver = TorchDiffEqSolver(method="dopri5", use_adjoint=False, rtol=1e-6, atol=1e-8)
    sim = Simulator(model, solver=solver)

    # Sin entradas externas: el control 'pd' cierra el lazo.
    dt = 1e-3
    T  = 2.0
    sim.simulate(dt=dt, total_time=T, externals_fn=None, reset_time=True)

    x = sim.states["mass"][:, 0:1]
    v = sim.states["mass"][:, 1:2]
    print(f"[hybrid torchdiffeq] t_final = {sim.t:.6f} s, k = {sim.k}")
    print("x(B) =", x.squeeze(-1).tolist())
    print("v(B) =", v.squeeze(-1).tolist())


if __name__ == "__main__":
    main()
