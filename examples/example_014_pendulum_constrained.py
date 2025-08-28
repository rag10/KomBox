
import os, sys, torch

# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from kombox.core.block import ContinuousBlock
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver
# Si quieres probar implícito + Baumgarte:
# check_implicit_baumgarte = True
# if check_implicit_baumgarte:
#     from kombox.core.algebraic.newton_krylov import NewtonKrylov
#     from kombox.core.solvers_trapezoidal import TrapezoidalSolver

class Mass2D(ContinuousBlock):
    """
    Masa puntual en 2D con gravedad (eje Y hacia abajo).
    Estado: [x, y, vx, vy]  (S = 4)
    Entradas: Fx, Fy (fuerzas externas)
    Salidas: x, y, vx, vy
    """
    def __init__(self, m: float = 1.0, g: float = 9.81):
        super().__init__()
        # Declaración unificada de IO + tamaño de estado
        self.declare_io(
            inputs={"Fx": 1, "Fy": 1},
            outputs={"x": 1, "y": 1, "vx": 1, "vy": 1},
            state_size=4,
        )
        # Aliases de estado para inicialización legible desde Model.initialize(...)
        self.state_alias = {"x": 0, "y": 1, "vx": 2, "vy": 3}
        # Parámetros simples (buffers escalares)
        self.register_buffer("m", torch.tensor(float(m)))
        self.register_buffer("g", torch.tensor(float(g)))

    def ode(self, state, inputs, t):
        # Desempaquetar estado
        x  = state[:, [0]]; y  = state[:, [1]]
        vx = state[:, [2]]; vy = state[:, [3]]

        # Entradas (el framework rellena con ceros si faltan en Fase A)
        Fx = inputs["Fx"]; Fy = inputs["Fy"]

        # Dinámica (aclara: ay += g porque Y es hacia abajo)
        ax = Fx / self.m
        ay = Fy / self.m + self.g

        dx = torch.cat([vx, vy, ax, ay], dim=1)
        outs = {"x": x, "y": y, "vx": vx, "vy": vy}
        return dx, outs

def build_pendulum(L: float = 1.0, m: float = 1.0, g: float = 9.81) -> Model:
    mdl = Model("pendulum_constrained")
    mdl.add_block("M", Mass2D(m=m, g=g))

    # Conectar externals estrictos a los puertos del bloque
    mdl.connect("Fx", "M.Fx")
    mdl.connect("Fy", "M.Fy")

    # Restricción holonómica: g1(x,y)=x^2+y^2-L^2=0
    def g1(t, states, inbuf, model, z):
        x = states["M"][:, 0:1]; y = states["M"][:, 1:2]
        return x*x + y*y - (L*L)

    # Restricción de tangencia: g2 = x*vx + y*vy = 0 (sin componente radial)
    def g2(t, states, inbuf, model, z):
        x = states["M"][:, 0:1]; y = states["M"][:, 1:2]
        vx = states["M"][:, 2:3]; vy = states["M"][:, 3:4]
        return x*vx + y*vy

    mdl.add_constraint_eq("circle", g1)
    mdl.add_constraint_eq("tangency", g2)

    mdl.build()
    return mdl

def externals_zero(B: int):
    """Fx=0, Fy=0 para cumplir la API de externals (la gravedad va dentro del bloque)."""
    def fn(t, k):
        z = torch.zeros((B, 1), dtype=torch.float32)
        return {"Fx": {"Fx": z}, "Fy": {"Fy": z}}
    return fn

def main():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")
    B = 1
    L = 1.0

    m = build_pendulum(L=L)

    # Estado inicial: ángulo 60º, velocidad tangencial nula
    theta0 = torch.tensor([60.0 * 3.141592653589793 / 180.0], dtype=torch.float32)
    x0 = (L * torch.sin(theta0)).view(1, 1)
    y0 = (-L * torch.cos(theta0)).view(1, 1)
    vx0 = torch.zeros_like(x0)
    vy0 = torch.zeros_like(y0)

    m.initialize(
        batch_size=B, device=device, dtype=torch.float32,
        initial_states={"M": {"x": x0, "y": y0, "vx": vx0, "vy": vy0}},
    )

    # Solver explícito + proyección (rápido y suficiente)
    sim = Simulator(m, solver=RK45Solver())
    sim.enable_constraint_projection(True, tol=1e-9, max_iter=3, damping=1e-5)

    # Si quieres probar implícito + Baumgarte (opcional):
    # if check_implicit_baumgarte:
        # alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=20)
        # trap = TrapezoidalSolver(algebraic_solver=alg, baumgarte_enabled=True,
        #                          baumgarte_alpha=2.0, baumgarte_beta=10.0)
        # sim = Simulator(m, solver=trap)
        # sim.enable_constraint_projection(True, tol=1e-9, max_iter=2, damping=1e-8)

    dt = 0.001
    T  = 6.0
    steps = int(T / dt)
    ext = externals_zero(B)

    xs, ys = [], []
    for _ in range(steps):
        sim.step(dt, externals_fn=ext)
        st = m.states["M"]
        xs.append(st[:, 0:1].detach().cpu())
        ys.append(st[:, 1:2].detach().cpu())

    X = torch.cat(xs, dim=0).view(-1).numpy()
    Y = torch.cat(ys, dim=0).view(-1).numpy()

    # Pequeño reporte
    import numpy as np
    r = np.sqrt(X*X + Y*Y)
    print(f"radio medio: {r.mean():.4f}  (target L={L})")
    print(f"violación max |x^2+y^2-L^2| ≈ {np.abs(r - L).max():.2e}")

    # --- Animación GIF opcional ---
    SAVE_GIF = True
    if SAVE_GIF:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation
            from matplotlib.patches import Circle

            fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
            ax.set_aspect('equal')
            ax.set_xlim(-1.2*L, 1.2*L)
            ax.set_ylim(-1.2*L, 1.2*L)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Péndulo con proyección pos-paso")

            ax.add_patch(Circle((0, 0), L, fill=False, lw=1.0, alpha=0.6))
            trace, = ax.plot([], [], lw=1.0, alpha=0.7)
            bob,   = ax.plot([], [], 'o', ms=6)

            def init():
                trace.set_data([], [])
                bob.set_data([], [])
                return trace, bob

            def animate(i):
                i = min(i, len(X)-1)
                trace.set_data(X[:i+1], Y[:i+1])
                bob.set_data([X[i]], [Y[i]])
                return trace, bob

            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=20, blit=True)
            out_path = "pendulum_constrained.gif"
            anim.save(out_path, writer="pillow", fps=30)
            print(f"GIF guardado en: {out_path}")
        except Exception as e:
            print(f"[AVISO] No se pudo crear el GIF (instala matplotlib/pillow): {e}")

if __name__ == "__main__":
    main()
