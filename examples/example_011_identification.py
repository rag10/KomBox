# kombox/examples/example_011_identification.py
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
    """
    Modelo MSD:
        Fs = -k (x - x0)
        Fd = -c v
        F  = Fs + Fd + Fext
        Masa: dx = v ; dv = F/m ; outs x,v
    """
    m = Model("msd_id")
    m.add_block("spring", Spring(k=40.0, x0=0.0))   # k se reajustará en entrenamiento
    m.add_block("damper", Damper(c=1.2))
    m.add_block("sum",    Adder(n_inputs=3, width=1)).alias_inputs({"Fs":"in1","Fd":"in2","Fext":"in3"}) \
                                                     .alias_outputs({"F":"out"})
    m.add_block("mass",   Mass1D(m=1.0))
    # Wiring
    m.connect("spring.F", "sum.Fs")
    m.connect("damper.F", "sum.Fd")
    m.connect("sum.F",    "mass.F")
    m.connect("mass.x",   "spring.x")
    m.connect("mass.v",   "damper.v")

    # Externos
    m.connect("Fext",     "sum.Fext")  # entrada de fuerza
    m.connect("mass.x",   "x")         # salida externa (opcional)
    m.build()
    return m


def make_step_force_fn(B: int, amps: torch.Tensor, t_step: float = 0.010):
    """
    externals_fn(t, k) que aplica un escalón en Fext de amplitud por-muestra 'amps' a partir de t_step.
    'amps' debe ser (B,1) en el device/dtype correctos.
    """
    def fn(t: float, _k: int):
        val = (t >= t_step)
        # broadcasting boolean -> float
        F = amps if val else torch.zeros_like(amps)
        return {"Fext": {"Fext": F}}
    return fn


@torch.no_grad()
def simulate_final_x(model: Model, solver: TorchDiffEqSolver, B: int, x0: torch.Tensor, v0: torch.Tensor,
                     amps: torch.Tensor, dt: float, T: float, device, dtype) -> torch.Tensor:
    """
    Simula y devuelve x(T) (B,1) sin construir grafo (para generar 'mediciones' verdad).
    """
    model.initialize(batch_size=B, device=device, dtype=dtype,
                     initial_states={"mass": {"x": x0, "v": v0}})
    sim = Simulator(model, solver=solver)
    ext_fn = make_step_force_fn(B, amps)
    sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, reset_time=True)
    return model.states["mass"][:, 0:1].clone()  # x(T)


def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")  # cambia a "cuda" si tienes GPU
    dtype  = torch.float32

    # ------------------ “Datos” (verdad) ------------------
    # Batch de experimentos con distintas amplitudes de fuerza
    B = 32
    amps_true = torch.linspace(0.5, 2.0, B, device=device, dtype=dtype).reshape(B,1)

    # Estados iniciales
    x0 = torch.zeros((B,1), device=device, dtype=dtype)
    v0 = torch.zeros((B,1), device=device, dtype=dtype)

    # Parámetros de simulación
    dt = 1e-3
    T  = 1.0

    # Modelo "verdadero" (solo para generar datos), con k_true
    k_true = 40.0
    model_true = build_msd()
    model_true.blocks["spring"].set_param("k", k_true)
    solver_data = TorchDiffEqSolver(method="dopri5", use_adjoint=False, rtol=1e-7, atol=1e-9)
    xT_true = simulate_final_x(model_true, solver_data, B, x0, v0, amps_true, dt, T, device, dtype)  # (B,1)

    # ------------------ Modelo a ajustar ------------------
    model = build_msd()
    # Inicializa k con un valor “malo” para que el ajuste tenga trabajo
    model.blocks["spring"].set_param("k", 15.0)              # (1,1)
    model.blocks["spring"].make_param_trainable("k", True)   # nn.Parameter en 'param_k'

    # Inicialización (ojo: expandirá (1,1) -> (B,1); entrenarás B copias de k iguales por simetría)
    model.initialize(batch_size=B, device=device, dtype=dtype,
                     initial_states={"mass": {"x": x0, "v": v0}})

    # Solver con adjoint para ahorrar memoria en horizontes largos
    solver = TorchDiffEqSolver(method="dopri5", use_adjoint=True, rtol=1e-6, atol=1e-8)
    sim = Simulator(model, solver=solver)

    # Optimización
    # Acceso directo al Parameter: se registra como atributo 'param_k' en el bloque spring
    k_param = model.blocks["spring"].param_k
    opt = torch.optim.Adam([k_param], lr=5e-2)

    # Congela damper/mass por si acaso (no necesario si quedaron como buffers)
    for p in model.blocks["damper"].parameters(): p.requires_grad_(False)
    for p in model.blocks["mass"].parameters():   p.requires_grad_(False)

    # La misma familia de fuerzas que usamos para generar los datos
    ext_fn = make_step_force_fn(B, amps_true)

    n_epochs = 80
    for epoch in range(1, n_epochs+1):
        # Re-inicializa estados (pero NO el valor de k, que es un Parameter vivo)
        model.initialize(batch_size=B, device=device, dtype=dtype,
                         initial_states={"mass": {"x": x0, "v": v0}})
        sim.reset_time()

        # Simula con gradientes (no @no_grad)
        sim.simulate(dt=dt, total_time=T, externals_fn=ext_fn, reset_time=False)

        # Pérdida = MSE en x(T)
        xT = model.states["mass"][:, 0:1]  # (B,1)
        loss = torch.mean((xT - xT_true)**2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Report
        k_mean = model.blocks["spring"].param_k.detach().mean().item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}/{n_epochs}] loss={loss.item():.6e}   k_mean={k_mean:.4f}")

    # Resultado final
    k_learned = model.blocks["spring"].param_k.detach()
    print("\n--- Resultado ---")
    print("k_true     =", float(k_true))
    print("k_est_mean =", float(k_learned.mean()))
    print("k_est_std  =", float(k_learned.std()))


if __name__ == "__main__":
    main()
