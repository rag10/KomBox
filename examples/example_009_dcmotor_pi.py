from __future__ import annotations
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import matplotlib.pyplot as plt

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver
from kombox.core.recorders import MemoryRecorder  # ⬅️ usa el recorder del core

from kombox.blocks.basic import Adder, Gain
from kombox.blocks.control import PID
from kombox.blocks.electrical import DCMotor

# Si tienes helpers de plotting:
try:
    from kombox.utils.plotting import plot_series_batch
    HAVE_HELPERS = True
except Exception:
    HAVE_HELPERS = False


def build_model() -> Model:
    m = Model("dcmotor_pi")

    m.add_block("motor", DCMotor(R=1.0, L=0.5, ke=0.01, kt=0.01, J=0.01, b=0.1))
    m.add_block("neg",   Gain(gain=-1.0))  # canónicos: 'in' → 'out'
    m.add_block("sum",   Adder(n_inputs=2, width=1)).alias_inputs({"r":"in1", "ym":"in2"}).alias_outputs({"e":"out"})
    m.add_block("pi",    PID(Kp=2.0, Ki=50.0, Kd=0.0, u_min=0.0, u_max=12.0))

    m.connect("motor.omega", "neg.in")
    m.connect("neg.out",     "sum.ym")

    m.connect("r",           "sum.r")
    m.connect("sum.e",       "pi.e")
    m.connect("pi.u",        "motor.V")
    m.connect("tau",         "motor.tau_load")

    m.build()
    return m


def make_externals(B: int):
    def fn(t: float, k: int):
        r = 50.0 if t >= 0.02 else 0.0         # ref ω
        tau = 0.02 if t >= 0.2 else 0.0        # par carga
        return {
            "r":   {"r":   torch.full((B,1), r)},
            "tau": {"tau_load": torch.full((B,1), tau)},
        }
    return fn


def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    m = build_model()
    B = 3
    # i0 y ω0 iniciales
    i0 = torch.zeros((B,1), device=device); w0 = torch.zeros((B,1), device=device)
    m.initialize(batch_size=B, device=device, dtype=torch.float32,
                 initial_states={"motor": {"i": i0, "omega": w0}})

    sim = Simulator(m, solver=RK45Solver())
    ext = make_externals(B)

    rec = MemoryRecorder(
        model=m,
        signals=[
            "motor.omega",  # (B,1)
            "motor.i",      # (B,1)
            "sum.e",        # (B,1)
            "pi.u",         # (B,1)
        ],
        # opcional: estados por alias/índice/slice
        # states={"motor": ["i", "omega"]},
        store_time=True,
        detach_to_cpu=True,
    )

    dt, T = 1e-4, 0.5
    sim.simulate(dt=dt, total_time=T, externals_fn=ext, recorder=rec, reset_time=True)

    packed = rec.stacked()
    t      = packed["t"]                     # (T,)
    omega  = packed["motor.omega"]           # (T,B,1)
    i      = packed["motor.i"]               # (T,B,1)
    e      = packed["sum.e"]                 # (T,B,1)
    u      = packed["pi.u"]                  # (T,B,1)

    # --------- Plots ---------
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    idx = 0  # muestra del batch a trazar

    if HAVE_HELPERS:
        plot_series_batch(axs[0,0], t, omega, idx=idx, label="omega [rad/s]")
        plot_series_batch(axs[0,1], t, i,     idx=idx, label="i [A]")
        plot_series_batch(axs[1,0], t, e,     idx=idx, label="e")
        plot_series_batch(axs[1,1], t, u,     idx=idx, label="u [V]")
    else:
        # fallback sencillo sin helpers
        axs[0,0].plot(t, omega[:, idx, 0]); axs[0,0].set_title("omega [rad/s]")
        axs[0,1].plot(t, i[:, idx, 0]);     axs[0,1].set_title("i [A]")
        axs[1,0].plot(t, e[:, idx, 0]);     axs[1,0].set_title("error")
        axs[1,1].plot(t, u[:, idx, 0]);     axs[1,1].set_title("u [V]")
        for ax in axs.flat: ax.grid(True, alpha=0.3)

    for ax in axs[-1,:]: ax.set_xlabel("t [s]")
    fig.suptitle("DC Motor + PI (muestra 0)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
