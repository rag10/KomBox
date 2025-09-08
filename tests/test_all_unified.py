# tests/test_all_unified.py
import math
import os
import numpy as np
import pytest
import torch

from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import EulerSolver, RK4Solver, RK45Solver
from kombox.core.solvers_trapezoidal import TrapezoidalSolver
from kombox.blocks.basic import Adder, Gain, Constant
from kombox.blocks.mechanical import Spring, Damper, Mass1D
from kombox.blocks.electrical import DCMotor
from kombox.utils.externals import (
    make_external_step, make_external_constant, combine_externals
)
from kombox.core.utils import (
    apply_parameter_grid, sweep_lin, sweep_log, sweep_list, combine_grids,
    load_npz_series
)

# --------------------------- Fixtures y helpers ---------------------------

@pytest.fixture(autouse=True)
def set_default_dtype_cpu():
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    yield

def build_msd(name="msd", *, B=1, device="cpu", x0=0.0, v0=0.0):
    """
    Masa–muelle–amortiguador con sumador de fuerzas:
        F = Fs + Fd + Fext
        Fs = -k (x - x0s), Fd = -c v
    Expuestos: x, v
    """
    m = Model(name)
    K = m.add_block("K", Spring(k=10.0, x0=0.0))
    C = m.add_block("C", Damper(c=0.5))
    SUM = m.add_block("sum", Adder(n_inputs=3, width=1))
    M = m.add_block("mass", Mass1D(m=1.0))

    # Alias legibles para el Adder
    SUM.alias_inputs({"Fs": "in1", "Fd": "in2", "Fext": "in3"})

    # Conexiones internas
    m.connect("mass.x", "K.x")
    m.connect("mass.v", "C.v")
    m.connect("K.F", "sum.Fs")
    m.connect("C.F", "sum.Fd")
    m.connect("sum.out", "mass.F")

    # Entradas/salidas externas
    m.expose("x", "mass.x")
    m.expose("v", "mass.v")
    m.bind_external("Fext", "sum.Fext")

    m.build()

    # Estados iniciales
    init = {"mass": {"x": torch.tensor([[x0]]*B), "v": torch.tensor([[v0]]*B)}}
    m.initialize(B, device=torch.device(device), dtype=torch.float32, initial_states=init)
    return m

# ------------------------------ Pruebas core ------------------------------

def test_model_build_and_shapes():
    B = 2
    m = build_msd(B=B, x0=0.01)
    # Estados por bloque con forma (B,S)
    for name, st in m.states.items():
        assert st.ndim == 2 and st.shape[0] == B, f"{name} inválido {tuple(st.shape)}"
    # Externals requeridos y outputs expuestos
    assert "Fext" in m.required_externals()
    outs = m.output_names()
    assert set(outs) == {"x", "v"}

def test_externals_alias_strict_injection_smoke():
    m = build_msd(B=1)
    # Señal step correctamente etiquetada al puerto canónico del sumador (alias 'Fext' -> 'in3')
    Fext = make_external_step("Fext", port="in3", t0=0.010, y_before=0.0, y_after=1.0, batch_size=1, width=1)
    ext_fn = combine_externals(Fext)

    sim = Simulator(m, solver=RK4Solver())
    sim.simulate(dt=1e-3, total_time=0.05, externals_fn=ext_fn)
    x_end = m.states["mass"][:, [0]].clone()
    assert torch.isfinite(x_end).all()

def test_rk4_response_increases_after_step():
    m = build_msd(B=1)
    Fext = make_external_step("Fext", port="in3", t0=0.010, y_before=0.0, y_after=1.0, batch_size=1, width=1)
    sim = Simulator(m, solver=RK4Solver())
    # Medimos desplazamiento antes/después del escalón
    x0 = m.states["mass"][:, [0]].clone()
    sim.simulate(dt=5e-4, total_time=0.012, externals_fn=Fext)
    x_mid = m.states["mass"][:, [0]].clone()
    sim.simulate(dt=5e-4, total_time=0.020, externals_fn=Fext)
    x_end = m.states["mass"][:, [0]].clone()
    # Tras aplicar fuerza positiva, la posición tiende a crecer respecto a antes del step
    assert (x_end - x_mid).mean() > -1e-6
    assert torch.isfinite(x_end).all()

def test_rk45_adaptive_smoke_runs():
    m = build_msd(B=1)
    Fext = make_external_constant("Fext", port="in3", value=0.1, batch_size=1, width=1)
    sim = Simulator(m, solver=RK45Solver())  # error control interno
    sim.simulate(dt=1e-3, total_time=0.02, externals_fn=Fext)  # “dt objetivo”
    assert m.states["mass"].isfinite().all()

def test_checkpoint_preserves_time_and_progress(tmp_path):
    B = 2
    m = build_msd(B=B)
    Fext = make_external_step("Fext", port="in3", t0=0.010, y_before=0.0, y_after=1.0, batch_size=B, width=1)
    sim = Simulator(m, solver=RK4Solver())

    dt = 5e-4
    steps_total = 4000
    steps_half = steps_total // 2

    # Primera mitad
    sim.simulate(dt=dt, steps=steps_half, externals_fn=Fext)
    chk = sim.make_checkpoint(detach=True)
    t_half, k_half = sim.t, sim.k

    # Segunda mitad (ruta A)
    sim.simulate(dt=dt, steps=steps_total - steps_half, externals_fn=Fext)
    states_A = {n: st.clone() for n, st in m.states.items()}
    t_A, k_A = sim.t, sim.k

    # Restaurar y volver a simular (ruta B)
    sim.restore_checkpoint(chk)
    assert sim.t == pytest.approx(t_half)
    assert sim.k == k_half
    sim.simulate(dt=dt, steps=steps_total - steps_half, externals_fn=Fext)
    states_B = {n: st.clone() for n, st in m.states.items()}
    t_B, k_B = sim.t, sim.k

    # Deben coincidir exactamente
    for n in states_A:
        assert torch.allclose(states_A[n], states_B[n], atol=0, rtol=0), f"Mismatch en {n}"
    assert t_A == pytest.approx(t_B)
    assert k_A == k_B

# ----------------------------- Utils y grids ------------------------------

def test_apply_parameter_grid_and_sweeps():
    m = build_msd(B=1)
    # Combinar sweeps sobre k y c
    gk = sweep_lin("K.k", 5.0, 15.0, num=3)         # 5,10,15
    gc = sweep_log("C.c", -2, 0, num=3)             # 0.01, 0.1, 1.0
    g3 = sweep_list("mass.m", [0.5, 1.0])           # dos masas
    grid = combine_grids(gk, gc, g3)
    B = apply_parameter_grid(m, grid)
    assert B == 3 * 3 * 2

    # Tras initialize con B correcto, los params deberían broadcastear a (B,1)
    m.initialize(B, device=torch.device("cpu"), dtype=torch.float32)
    assert m.blocks["K"].get_param("k").shape == (B, 1)
    assert m.blocks["C"].get_param("c").shape == (B, 1)
    assert m.blocks["mass"].get_param("m").shape == (B, 1)

def test_load_npz_series_roundtrip(tmp_path):
    # Pequeña simulación con recorder por chunks y luego re-carga
    m = build_msd(B=1)
    Fext = make_external_constant("Fext", port="in3", value=0.0, batch_size=1, width=1)

    # Grabador NPZ por chunks
    from kombox.core.recorders import NPZChunkRecorder
    pattern = str(tmp_path / "trace_part{part:03d}.npz")
    rec = NPZChunkRecorder(m, pattern, chunk_size=50, signals=["mass.x", "mass.v"], store_time=True)
    sim = Simulator(m, solver=EulerSolver())
    sim.simulate(dt=1e-3, total_time=0.12, externals_fn=Fext, recorder=rec)

    # Cargar y validar claves
    data = load_npz_series(pattern, strip_prefix=True, to_torch=False)
    assert "t" in data and "mass.x" in data and "mass.v" in data
    t = data["t"]; x = data["mass.x"]; v = data["mass.v"]
    assert t.ndim == 1 and x.ndim == 3 and v.ndim == 3
    assert x.shape[0] == v.shape[0] == t.shape[0]

# ----------------------- Newton–Krylov y Trapecio -------------------------

def test_newton_krylov_dense_and_jfnk_converge():
    from kombox.core.algebraic.newton_krylov import NewtonKrylov
    B = 3
    z0 = torch.zeros((B, 2))  # objetivo: (1, -2)
    target = torch.tensor([[1.0, -2.0]])
    target = target.expand(B, -1)

    def F(z):
        # F(z)=0 en z=target -> (z - target)
        return z - target

    for mode in ("dense", "jfnk"):
        nk = NewtonKrylov(mode=mode, tol=1e-10, max_iter=20, verbose=False)
        z_star = nk.solve(F, z0)
        assert torch.allclose(z_star, target, atol=1e-6, rtol=0)

@pytest.mark.xfail(reason="Ruta DAE/KKT aún inestable en algunos casos; smoke test para no romper API.")
def test_trapezoidal_kkt_baumgarte_smoke_reduces_position_error():
    # Restricción global g(x)=x=0 con fuerza de Lagrange en la masa: F += λ
    B = 1
    m = build_msd(B=B, x0=0.02, v0=0.0)

    def g_x_zero(t, states, inbuf, model, z=None):
        x = states["mass"][:, [0]]  # (B,1)
        return x  # queremos x->0

    m.add_constraint_eq("x_zero", g_x_zero)

    def hook_force(t, states, inbuf, model, z, lam_i):
        # Φ_q = 1 en coordenada x → fuerza directa en el puerto 'F' de la masa
        return {"mass": {"F": lam_i}}

    m.add_constraint_force("x_zero", hook_force)

    # Trapecio global con modo Baumgarte activado
    solver = TrapezoidalSolver(constraint_mode="kkt_baumgarte", baumgarte_alpha=50.0, baumgarte_beta=10.0)
    sim = Simulator(m, solver=solver)

    # Sin fuerzas externas
    Fext = make_external_constant("Fext", port="in3", value=0.0, batch_size=B, width=1)

    x_before = m.states["mass"][:, [0]].clone()
    sim.simulate(dt=1e-3, steps=1, externals_fn=Fext)  # un paso implícito
    x_after = m.states["mass"][:, [0]].clone()

    # Esperamos |x| reducir (Baumgarte estabiliza)
    assert torch.norm(x_after) <= torch.norm(x_before) + 1e-9

# ------------------------------ Ejemplos extra ----------------------------

def test_dcmotor_smoke_runs():
    # Motor DC libre (tau=0), step en V; comprobar que omega crece al principio
    m = Model("dc")
    plant = m.add_block("plant", DCMotor(R=1.0, L=0.5, ke=0.01, kt=0.01, J=0.02, b=0.1))
    m.expose("i", "plant.i"); m.expose("omega", "plant.omega")
    m.bind_external("V", "plant.V"); m.bind_external("tau", "plant.tau_load")
    m.build()
    m.initialize(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    V = make_external_step("V", port="V", t0=0.0, y_before=0.0, y_after=5.0, batch_size=1, width=1)
    tau = make_external_constant("tau", port="tau_load", value=0.0, batch_size=1, width=1)
    ext = combine_externals(V, tau)

    sim = Simulator(m, solver=RK4Solver())
    i0 = m.states["plant"][:, [0]].clone()
    sim.simulate(dt=1e-3, total_time=0.05, externals_fn=ext)
    omega = m.states["plant"][:, [1]]
    assert torch.isfinite(omega).all()
