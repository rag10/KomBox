from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver
from kombox.blocks.mechanical import Mass1D, Spring, Damper
from kombox.blocks.basic import Adder, Gain


def build_model() -> Model:
    """
    Planta MSD con control proporcional u = Kp * (r - x)
    Estructura:
      Fs = -k(x - x0)
      Fd = -c v
      F  = Fs + Fd + u
      Mass: dx=v, dv=F/m
      e = r + (-x)
      u = Kp * e
    """
    m = Model("msd_opt_kp")

    # Bloques planta
    m.add_block("spring", Spring(k=40.0, x0=0.0))
    m.add_block("damper", Damper(c=1.2))
    m.add_block("mass",   Mass1D(m=1.0))
    m.add_block("sumF",   Adder(n_inputs=3, width=1)).alias_inputs({"Fs":"in1","Fd":"in2","u":"in3"}).alias_outputs({"F":"out"})

    # Bloques control
    m.add_block("negx",   Gain(gain=-1.0))                       # y = -x
    m.add_block("sumE",   Adder(n_inputs=2, width=1)).alias_inputs({"r":"in1","x":"in2"}).alias_outputs({"e":"out"})
    m.add_block("K",      Gain(gain=0.5))                        # Kp inicial (entrenable)

    # Cableado planta
    m.connect("spring.F", "sumF.Fs")
    m.connect("damper.F", "sumF.Fd")
    m.connect("sumF.F",   "mass.F")
    m.connect("mass.x",   "spring.x")
    m.connect("mass.v",   "damper.v")

    # Cableado control
    m.connect("mass.x",   "negx.in")
    m.connect("negx.out", "sumE.x")
    m.connect("r",        "sumE.r")    # referencia externa
    m.connect("sumE.e",   "K.in")
    m.connect("K.out",    "sumF.u")

    # (opcional) salida externa
    m.connect("mass.x",   "x")

    m.build()
    return m


def make_ref_fn(B: int, r_final: float = 0.05, t_step: float = 0.0):
    """Referencia por escalón r(t)=r_final desde t_step."""
    def fn(t: float, k: int):
        val = r_final if t >= t_step else 0.0
        return {"r": {"r": torch.full((B,1), float(val))}}
    return fn


def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    # 1) Modelo y solver
    model = build_model()
    B = 1
    x0 = torch.zeros((B,1), device=device)
    v0 = torch.zeros((B,1), device=device)

    # Hacemos entrenable el Kp del bloque Gain "K"
    model.blocks["K"].make_param_trainable("gain", True)
    # Inicializamos estados
    model.initialize(batch_size=B, device=device, dtype=torch.float32,
                     initial_states={"mass": {"x": x0, "v": v0}})

    sim = Simulator(model, solver=RK45Solver())

    # 2) Configuración de simulación
    dt, T = 1e-3, 1.0
    ref_fn = make_ref_fn(B, r_final=0.05)  # referencia 5 cm

    # 3) Objetivo: minimizar (x(T) - r)^2
    kp_param = model.blocks["K"].param_gain  # nn.Parameter shape (1,1)
    opt = torch.optim.Adam([kp_param], lr=0.2)

    n_epochs = 40
    for epoch in range(1, n_epochs+1):
        # Re-inicializa estados (no toca el valor de Kp)
        model.initialize(batch_size=B, device=device, dtype=torch.float32,
                         initial_states={"mass": {"x": x0, "v": v0}})
        sim.reset_time()

        # Simulación (forward con gradiente)
        sim.simulate(dt=dt, total_time=T, externals_fn=ref_fn, reset_time=False)

        # Error final
        xT = model.states["mass"][:, 0:1]               # (B,1)
        rT = torch.full_like(xT, 0.05)                  # r(T)=0.05
        loss = torch.mean((xT - rT) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{epoch:02d}/{n_epochs}] loss={loss.item():.6e}  Kp={float(kp_param.detach().mean()):.4f}")

    print("\nResultado:")
    print("Kp aprendido =", float(kp_param.detach().mean()))
    print("x(T) =", float(model.states["mass"][:,0:1].detach().mean()))
    print("r(T) = 0.05")


if __name__ == "__main__":
    main()
