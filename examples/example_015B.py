#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example F3 — Trapezoidal con restricciones (KomBox) con progreso en consola
Modos: KKT, Baumgarte y Mixto.

- KKT:     masa 1D con g(x)=x=0 (λ por hooks)
- BG:      misma masa con estabilización de Baumgarte (usa hooks para λ_bg)
- Mixto:   dos masas: g1(x1)=0 (λ) y g2(x2)-x_target=0 (BG)

Se comparan las trayectorias y se guardan gráficas PNG. Muestra progreso
cada cierto número de pasos (configurable) y métricas útiles por consola.
"""

from __future__ import annotations
import os, sys, torch
# permitir ejecutar desde /examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import matplotlib.pyplot as plt

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.blocks.mechanical import Mass1D
from kombox.core.algebraic.newton_krylov import NewtonKrylov
from kombox.core.solvers_trapezoidal import TrapezoidalSolver


# ------------------------ util plotting ------------------------
def _savefig(fig, fname):
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"[saved] {fname}")


def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()


# ------------------------ util progreso ------------------------
def _progress(k, steps, t, label, metrics: dict | None = None, final: bool = False):
    """
    Imprime progreso en formato compacto: [LABEL] k/steps (pct%) t=... | m1=... | ...
    """
    pct = int((k / max(1, steps)) * 100)
    msg = f"[{label}] {k:4d}/{steps:<4d} ({pct:3d}%)  t={t:.5f}"
    if metrics:
        for name, val in metrics.items():
            try:
                val = float(val)
            except Exception:
                pass
            msg += f" | {name}={val:.3e}"
    end = "\n" if final else "\r"
    print(msg, end=end, flush=True)


# -------------------------- KKT (1 masa) --------------------------
def run_kkt(B=3, dt=0.02, steps=120, *, progress_every=None, device="cpu", dtype=torch.float32, verbose=True):
    torch.set_default_dtype(dtype)
    device = torch.device(device)

    m = Model("kkt_1d")
    m.add_block("mass", Mass1D(m=1.0))

    # Puerto de fuerza externo
    m.connect("F", "mass.F")

    # Restricción global g(x)=x=0
    def g_pos_zero(t, states, inbuf, model, z):
        return states["mass"][:, 0:1]

    m.add_constraint_eq("x_equals_0", g_pos_zero)

    # Hook de fuerza: Phi_q^T * λ = 1 * λ
    def hook_force(t, states, inbuf, model, z, lam_i):
        return {"mass": {"F": lam_i}}

    m.add_constraint_force("x_equals_0", hook_force)
    m.build()

    # Estado inicial (x!=0 para ver corrección), v=0
    x0 = torch.tensor([[0.2], [-0.1], [0.05]], dtype=dtype)
    v0 = torch.zeros((B, 1), dtype=dtype)
    m.initialize(batch_size=B, device=device, dtype=dtype, initial_states={"mass": {"x": x0, "v": v0}})

    # Externals: F=0
    def ext_zero(t, k):
        return {"F": {"F": torch.zeros((B, 1), dtype=dtype, device=device)}}

    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30)
    trap = TrapezoidalSolver(algebraic_solver=alg, constraint_mode="kkt")
    sim = Simulator(m, solver=trap)
    sim.enable_constraint_projection(False)

    if progress_every is None:
        progress_every = max(1, steps // 20)  # ~5%

    T = dt * steps
    ts = []
    xs = []
    vs = []
    lams = []

    for k in range(steps):
        # logs previos al step
        g = m.states["mass"][:, 0:1]
        if verbose and (k % progress_every == 0 or k == steps - 1):
            _progress(
                k,
                steps,
                sim.t,
                "KKT",
                metrics={"max|g|": g.abs().max(), "max|x|": m.states["mass"][:, 0:1].abs().max()},
                final=(k == steps - 1),
            )

        ts.append(sim.t)
        xs.append(m.states["mass"][:, 0:1].clone())
        vs.append(m.states["mass"][:, 1:2].clone()) if m.states["mass"].shape[1] > 1 else vs.append(
            torch.zeros_like(xs[-1])
        )

        sim.step(dt, externals_fn=ext_zero)

        # Multiplicador (opcional) desde el último NK
        lam = getattr(alg, "last_solution", torch.zeros((B, 0), device=device))
        lams.append(lam if lam.numel() else torch.zeros((B, 1), device=device))

    ts.append(T)
    xs.append(m.states["mass"][:, 0:1].clone())
    vs.append(m.states["mass"][:, 1:2].clone()) if m.states["mass"].shape[1] > 1 else vs.append(
        torch.zeros_like(xs[-1])
    )

    return (
        _to_numpy(torch.tensor(ts)),
        _to_numpy(torch.cat(xs, dim=0)),
        _to_numpy(torch.cat(vs, dim=0)),
        _to_numpy(torch.cat(lams, dim=0)[:, :1]),
    )


# ---------------------- Baumgarte (1 masa) ----------------------
def run_baumgarte(
    B=3,
    dt=0.02,
    steps=120,
    *,
    alpha=1.0,
    beta=10.0,
    progress_every=None,
    device="cpu",
    dtype=torch.float32,
    verbose=True,
):
    torch.set_default_dtype(dtype)
    device = torch.device(device)

    m = Model("bg_1d")
    m.add_block("mass", Mass1D(m=1.0))
    m.connect("F", "mass.F")

    # g(x)=x=0
    def g_pos_zero(t, states, inbuf, model, z):
        return states["mass"][:, 0:1]

    m.add_constraint_eq("x_equals_0", g_pos_zero)

    # Hook de fuerza para aplicar λ_bg a 'mass.F'
    def hook_force(t, states, inbuf, model, z, lam_i):
        return {"mass": {"F": lam_i}}

    m.add_constraint_force("x_equals_0", hook_force)
    m.build()

    # Estado inicial
    x0 = torch.tensor([[0.2], [-0.1], [0.05]], dtype=dtype)
    v0 = torch.zeros((B, 1), dtype=dtype)
    m.initialize(batch_size=B, device=device, dtype=dtype, initial_states={"mass": {"x": x0, "v": v0}})

    def ext_zero(t, k):
        return {"F": {"F": torch.zeros((B, 1), dtype=dtype, device=device)}}

    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30)
    trap = TrapezoidalSolver(
        algebraic_solver=alg,
        constraint_mode="kkt_baumgarte",
        baumgarte_alpha=alpha,
        baumgarte_beta=beta,
    )
    sim = Simulator(m, solver=trap)
    sim.enable_constraint_projection(False)

    if progress_every is None:
        progress_every = max(1, steps // 20)

    T = dt * steps
    ts, xs, vs = [], [], []
    for k in range(steps):
        # logs previos
        g = m.states["mass"][:, 0:1]
        if verbose and (k % progress_every == 0 or k == steps - 1):
            _progress(
                k,
                steps,
                sim.t,
                "BG ",
                metrics={"max|g|": g.abs().max(), "max|x|": m.states["mass"][:, 0:1].abs().max()},
                final=(k == steps - 1),
            )

        ts.append(sim.t)
        xs.append(m.states["mass"][:, 0:1].clone())
        vs.append(m.states["mass"][:, 1:2].clone()) if m.states["mass"].shape[1] > 1 else vs.append(
            torch.zeros_like(xs[-1])
        )
        sim.step(dt, externals_fn=ext_zero)

    ts.append(T)
    xs.append(m.states["mass"][:, 0:1].clone())
    vs.append(m.states["mass"][:, 1:2].clone()) if m.states["mass"].shape[1] > 1 else vs.append(
        torch.zeros_like(xs[-1])
    )

    return _to_numpy(torch.tensor(ts)), _to_numpy(torch.cat(xs, dim=0)), _to_numpy(torch.cat(vs, dim=0))


# -------------------------- Mixto (2 masas) --------------------------
def run_mixed(
    B=3,
    dt=0.02,
    steps=120,
    *,
    alpha=1.0,
    beta=10.0,
    x2_target=0.1,
    progress_every=None,
    device="cpu",
    dtype=torch.float32,
    verbose=True,
):
    torch.set_default_dtype(dtype)
    device = torch.device(device)

    m = Model("mixed_2m")
    m.add_block("m1", Mass1D(m=1.0))
    m.add_block("m2", Mass1D(m=1.0))
    m.connect("F1", "m1.F")
    m.connect("F2", "m2.F")

    # g1(x1)=x1=0 (KKT)
    def g1_x1_zero(t, states, inbuf, model, z):
        return states["m1"][:, 0:1]

    # g2(x2)=x2 - x_target = 0 (Baumgarte)
    def g2_x2_target(t, states, inbuf, model, z):
        return states["m2"][:, 0:1] - x2_target

    m.add_constraint_eq("g1_x1_zero", g1_x1_zero)
    m.add_constraint_eq("g2_x2_target", g2_x2_target)

    # Hooks de fuerza para cada restricción
    def hook1(t, states, inbuf, model, z, lam_i):
        return {"m1": {"F": lam_i}}

    def hook2(t, states, inbuf, model, z, lam_i):
        return {"m2": {"F": lam_i}}

    m.add_constraint_force("g1_x1_zero", hook1)
    m.add_constraint_force("g2_x2_target", hook2)
    m.build()

    # Estados iniciales
    x10 = torch.tensor([[0.15], [0.10], [0.05]], dtype=dtype)
    v10 = torch.zeros((B, 1), dtype=dtype)
    x20 = torch.tensor([[-0.20], [-0.10], [-0.05]], dtype=dtype)
    v20 = torch.zeros((B, 1), dtype=dtype)

    init_states = {"m1": {"x": x10, "v": v10}, "m2": {"x": x20, "v": v20}}
    m.initialize(batch_size=B, device=device, dtype=dtype, initial_states=init_states)

    def ext_zero(t, k):
        z = torch.zeros((B, 1), dtype=dtype, device=device)
        return {"F1": {"F": z}, "F2": {"F": z}}

    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=30)
    trap = TrapezoidalSolver(
        algebraic_solver=alg,
        constraint_mode="kkt_mixed",
        lam_mask=[True, False],  # g1 con λ (KKT), g2 con BG
        baumgarte_alpha=alpha,
        baumgarte_beta=beta,
    )
    sim = Simulator(m, solver=trap)
    sim.enable_constraint_projection(False)

    if progress_every is None:
        progress_every = max(1, steps // 20)

    T = dt * steps
    ts, x1s, x2s = [], [], []
    for k in range(steps):
        # logs previos
        g1 = m.states["m1"][:, 0:1]
        g2 = m.states["m2"][:, 0:1] - x2_target
        if verbose and (k % progress_every == 0 or k == steps - 1):
            _progress(
                k,
                steps,
                sim.t,
                "MIX",
                metrics={"max|g1|": g1.abs().max(), "max|g2|": g2.abs().max()},
                final=(k == steps - 1),
            )

        ts.append(sim.t)
        x1s.append(m.states["m1"][:, 0:1].clone())
        x2s.append(m.states["m2"][:, 0:1].clone())
        sim.step(dt, externals_fn=ext_zero)
    ts.append(T)
    x1s.append(m.states["m1"][:, 0:1].clone())
    x2s.append(m.states["m2"][:, 0:1].clone())

    return (
        _to_numpy(torch.tensor(ts)),
        _to_numpy(torch.cat(x1s, dim=0)),
        _to_numpy(torch.cat(x2s, dim=0)),
        float(x2_target),
    )


# ------------------------------- main -------------------------------
def main():
    out_dir = os.getcwd()

    # Parámetros
    B = 3
    dt = 0.02
    steps = 120
    alpha = 1.0
    beta = 10.0
    x2_target = 0.1
    progress_every = max(1, steps // 20)  # imprime cada ~5% de progreso

    # Ejecutar tres variantes (con progreso)
    t_kkt, x_kkt, v_kkt, lam_kkt = run_kkt(B=B, dt=dt, steps=steps, progress_every=progress_every, verbose=True)
    t_bg, x_bg, v_bg = run_baumgarte(
        B=B, dt=dt, steps=steps, alpha=alpha, beta=beta, progress_every=progress_every, verbose=True
    )
    t_mx, x1_mx, x2_mx, xt = run_mixed(
        B=B, dt=dt, steps=steps, alpha=alpha, beta=beta, x2_target=x2_target, progress_every=progress_every, verbose=True
    )

    # Métricas (máximo en batch)
    kkt_xT = float(abs(x_kkt[-1]).max())
    bg_xT = float(abs(x_bg[-1]).max())
    mx_x1T = float(abs(x1_mx[-1]).max())
    mx_x2_errT = float(abs(x2_mx[-1] - xt).max())

    print("\n=== Métricas finales (máximo en batch) ===")
    print(f"KKT  |x(T)|         = {kkt_xT:.6e}")
    print(f"BG   |x(T)|         = {bg_xT:.6e}")
    print(f"Mix  |x1(T)| (KKT)  = {mx_x1T:.6e}")
    print(f"Mix  |x2(T)-xt| (BG)= {mx_x2_errT:.6e}")

    # Para visualizar, tomamos la primera muestra del batch
    b = 0
    # 1) KKT vs BG: posición
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(t_kkt, x_kkt[:, b], label="KKT: x(t)")
    ax1.plot(t_bg, x_bg[:, b], label="Baumgarte: x(t)")
    ax1.set_title("Masa 1D — KKT vs Baumgarte (posición)")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("x [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    _savefig(fig1, os.path.join(out_dir, "f3_kkt_vs_baumgarte_pos.png"))

    # 2) KKT vs BG: velocidad
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(t_kkt, v_kkt[:, b], label="KKT: v(t)")
    ax2.plot(t_bg, v_bg[:, b], label="Baumgarte: v(t)")
    ax2.set_title("Masa 1D — KKT vs Baumgarte (velocidad)")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("v [m/s]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    _savefig(fig2, os.path.join(out_dir, "f3_kkt_vs_baumgarte_vel.png"))

    # 3) Mixto: x1 y x2 -> xtarget
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(t_mx, x1_mx[:, b], label="Mixto: x1(t) [KKT → 0]")
    ax3.plot(t_mx, x2_mx[:, b], label="Mixto: x2(t) [BG → x_target]")
    ax3.axhline(xt, linestyle="--", label="x_target")
    ax3.set_title("Modo mixto — dos masas")
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("x [m]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    _savefig(fig3, os.path.join(out_dir, "f3_mixed_pos.png"))

    print("\nFiguras guardadas en:", out_dir)
    print(" - f3_kkt_vs_baumgarte_pos.png")
    print(" - f3_kkt_vs_baumgarte_vel.png")
    print(" - f3_mixed_pos.png")


if __name__ == "__main__":
    main()
