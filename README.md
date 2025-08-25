# KomBox

KomBox es un framework de **modelado por bloques** en **PyTorch** con **simulación diferenciable** (autograd). Está pensado para identificación de sistemas, control y co-diseño, manteniendo un núcleo simple, extensible y “autograd-first”.

**Versión:** v0.1.0  
**Requisitos:** Python ≥ 3.10, PyTorch ≥ 2.1

---

## Características

- **Bloques** con y sin estado, conexiones explícitas y evaluación en **orden topológico**.
- **Simulación diferenciable**: todo el grafo es compatible con `loss.backward()` (no se usan `detach()` en rutas de cálculo).
- **Solvers**:
  - **No globales (por bloque)**: integración explícita clase A→B (p. ej. RK45).
  - **Globales (monolíticos)**: ODE en un solo sistema (p. ej. TorchDiffEq).
- **Externals estrictos con alias**: la inyección requiere el **nombre canónico** del puerto o un **alias** declarado en el bloque destino.
- **API DAE mínima** (opt-in): puntos de extensión para lazos algebraicos y restricciones.
- **Progreso opcional** en `Simulator.simulate` (impresión cada ~1 s).

---

## Instalación

Modo editable (recomendado durante desarrollo):

```bash
pip install -e .
```

Extras opcionales:

```bash
# Solvers ODE globales basados en torchdiffeq
pip install -e ".[dae]"

# Herramientas de desarrollo
pip install -e ".[dev]"
```

---

## Conceptos clave

### Fase A / Fase B
- **Fase A (outputs)**: se evalúan todos los bloques (con y sin estado) en **orden topológico de feedthrough** para rellenar entradas aguas abajo.
- **Fase B (estados)**: los bloques con estado actualizan su estado con el **solver** elegido.

### Solvers
- **No global** (por-bloque): el simulador recorre bloques con estado (Euler/RK4/RK45…).
- **Global** (monolítico): una única función/solver avanza todos los estados a la vez (TorchDiffEq).

---

## Externals (entradas) — **modo estricto con alias**

Cuando conectas una entrada externa:

```python
m.connect("Fext", "sum.Fext")     # 'Fext' es el nombre de la señal externa
m.blocks["sum"].alias_inputs({"Fext":"in3", "Fs":"in1", "Fd":"in2"})
```

La conexión se normaliza al **puerto canónico** (`in3`). El inyector acepta:

- **canónico**: `{"Fext": {"in3": tensor}}`
- **alias**:    `{"Fext": {"Fext": tensor}}`

> Si dos destinos deben recibir valores **distintos**, usa **externals diferentes** (p. ej. `F1` y `F2`).

**Helpers** (`kombox/externals.py`) — siempre requieren `port`:

```python
from kombox.externals import make_external_step, make_external_constant, combine_externals

# MSD: el sumador mapea Fext -> in3
Fext = make_external_step("Fext", port="in3", t0=0.010, y_before=0.0, y_after=1.0,
                          batch_size=B, width=1, device=device, dtype=torch.float32)
ext_fn = combine_externals(Fext)
```

---

## API DAE mínima (opt-in)

- En bloques: `algebraic_residual(self, t, state, inputs, params) -> dict|Tensor` (por defecto vacío).
- En el modelo: `add_constraint_eq(name, fn)` y `build_residual(...)` para ensamblar restricciones globales.

Esto prepara el terreno para solvers implícitos (p. ej. Trapecio + Newton-Krylov).

---

## Uso rápido (esquema)

```python
from kombox.core.model import Model
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver

m = Model("demo")
# m.add_block(...); m.connect("A.y", "B.u"); ...
m.build()
m.initialize(batch_size=1, device="cpu")

sim = Simulator(m, solver=RK45Solver())
sim.simulate(dt=1e-3, total_time=1.0, externals_fn=None, progress=True)
```

---

## Ejemplos

```bash
python kombox/examples/example_001_msd.py        # masa-muelle-amortiguador
python kombox/examples/example_009_dcmotor_pi.py # motor DC con PI
```

- MSD: `{"Fext": {"in3": tensor}}` (o alias `{"Fext": {"Fext": tensor}}`).
- Motor DC: `{"tau": {"tau_load": tensor}}`.

---

## Tests

```bash
pytest -q
```

- Los tests de `torchdiffeq` se **saltan** si no está instalado.
- Suite de smoke e integración para rutas por-bloque y globales, externals estrictos, checkpoint/restore, y DAE mínima.

---

## Roadmap (siguiente versión)

- SCC + residual global para lazos algebraicos.
- Solver implícito **Trapezoidal** acoplado (x,z) + **Newton-Krylov**.
- Inicialización consistente de DAEs.
- Estabilización (Baumgarte) y proyección pos-paso.

---

## Licencia

MIT. Consulta `LICENSE`.
