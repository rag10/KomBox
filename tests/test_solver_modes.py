import torch
from kombox.core.model import Model
from kombox.core.block import ContinuousBlock
from kombox.core.simulator import Simulator
from kombox.core.solvers import SolverBase


class UnitIntegrator(ContinuousBlock):
    """dx/dt = u ; y = x."""
    def __init__(self, n=1):
        super().__init__()
        self.declare_io(inputs={"u": n}, outputs={"y": n})
        self._n = n
    def state_size(self): return self._n
    def init_state(self, B, device=None, dtype=None):
        return torch.zeros((B, self._n), device=device, dtype=dtype)
    def ode(self, state, inputs, t):
        dx = inputs["u"]
        return dx, {"y": state}


class ConstSource(ContinuousBlock):
    """Bloque sin estado que emite una constante (por dimensión)."""
    def __init__(self, n, value=1.0):
        super().__init__()
        self.declare_io(inputs={}, outputs={"y": n})
        self.value = float(value)
        self._n = n
    def state_size(self): return 0
    def init_state(self, B, device=None, dtype=None):
        return torch.zeros((B,0), device=device, dtype=dtype)
    def ode(self, state, inputs, t):
        y = torch.full((state.shape[0], self._n), self.value, device=state.device, dtype=state.dtype)
        return torch.zeros_like(state), {"y": y}


class DummyLocalSolver(SolverBase):
    """No global: ruta A→B por bloque."""
    name = "dummy-local"; is_global = False
    def __init__(self): super().__init__(); self.calls_cont = 0
    def step_continuous(self, block, state, inputs, dt, t):
        self.calls_cont += 1
        dx, outs = block.ode(state, inputs, t)
        return state + dt*dx, outs


class DummyGlobalSolver(SolverBase):
    """Global: avanza todo el modelo de una vez, sin externals en este test."""
    name = "dummy-global"; is_global = True
    def __init__(self): super().__init__(); self.calls_all = 0
    def step_all(self, model, states, dt, t, externals_time_fn=None):
        self.calls_all += 1
        # Propagar A: calcular entradas vía wiring usando outputs actuales
        # (versión mínima: sólo usamos la ConstSource → UnitIntegrator)
        # Construimos inputs para cada bloque:
        inbuf = {n: {} for n in model.blocks.keys()}
        # Emitir fuente
        src_out = model.blocks["C"]._expose_outputs(states.get("C", torch.zeros((next(iter(states.values())).shape[0],0))), {}, t)["y"]
        inbuf["G"]["u"] = src_out
        # Avance de estados del integrador
        x = states["G"]
        dx, o = model.blocks["G"].ode(x, inbuf["G"], t)
        new_states = dict(states)
        new_states["G"] = x + dt*dx
        # Salida del integrador
        outs = {"G": o, "C": {"y": src_out}}
        return new_states, outs


def make_model_local(B=2, n=1):
    m = Model("modes_local")
    m.add_block("G", UnitIntegrator(n=n))
    m.connect("u", "G.u")  # externals en ruta local
    m.build()
    m.initialize(batch_size=B, device=torch.device("cpu"), dtype=torch.float32)
    return m

def make_model_global(B=1, n=3, val=2.0):
    m = Model("modes_global")
    m.add_block("C", ConstSource(n=n, value=val))
    m.add_block("G", UnitIntegrator(n=n))
    m.connect("C.y", "G.u")
    m.build()
    m.initialize(batch_size=B, device=torch.device("cpu"), dtype=torch.float32)
    return m

def externals_const(u_val, B=2, n=1):
    def fn(t, k):
        return {"u": {"u": torch.full((B, n), float(u_val))}}
    return fn

def test_non_global_solver_path():
    m = make_model_local(B=2, n=1)
    sim = Simulator(m, solver=DummyLocalSolver())
    y0 = m.blocks["G"]._expose_outputs(m.states["G"], {"u": torch.zeros((2,1))}, t=0.0)["y"].clone()
    sim.step(0.1, externals_fn=externals_const(1.0, B=2, n=1))
    assert sim.solver.calls_cont == 1
    y1 = sim.model.states["G"]
    assert torch.allclose(y1 - y0, torch.full_like(y1, 0.1))

def test_global_solver_path():
    m = make_model_global(B=1, n=3, val=2.0)
    sim = Simulator(m, solver=DummyGlobalSolver())
    sim.step(0.2)  # sin externals
    assert sim.solver.calls_all == 1
    x = sim.model.states["G"]
    # x <- x + dt * u = 0 + 0.2 * 2 = 0.4
    assert torch.allclose(x, torch.full((1,3), 0.4))
