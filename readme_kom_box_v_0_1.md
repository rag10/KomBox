# KomBox

KomBox es un framework de modelado por bloques en PyTorch con simulación diferenciable (autograd), pensado para identificación, control y co-diseño. Soporta integración explícita por-bloque y solvers globales (monolíticos), y ahora incorpora una API mínima para DAEs/lazos algebraicos.

**Versión:** v0.1.0  
**Requisitos:** Python ≥ 3.10, PyTorch ≥ 2.1

---

## Novedades (v0.1.0)

- **Externals “estrictos con alias”**: la inyección de entradas externas exige el **nombre canónico del puerto** o uno de sus **alias**. Si no coincide, se lanza `KeyError` con mensaje claro (antes había un “broadcast laxo”).
- **Helpers de externals simplificados**: las *factories* (`make_external_*`) ahora **siempre requieren `port`** (canónico o alias). Más simple y predecible.
- **Solvers globales vs no globales**: documentación y tests de ambas rutas. `TorchDiffEqSolver` usa inyector estricto con alias; los solvers por-bloque siguen la ruta A→B.
- **API DAE mínima** (MVP, opt-in): `Block.algebraic_residual(...)`, `Model.add_constraint_eq(...)`, `Model.build_residual(...)`.
- **Progreso opcional** en `Simulator.simulate` (impresión cada ~1s).
- **Ejemplos y tests** actualizados; suite estable con `pytest`.

---

## Instalación

Editable (recomendado durante desarrollo):
```bash
pip install -e .
```

Extras opcionales:
```bash
# Solvers globales basados en torchdiffeq
pip install -e ".[dae]"

# Desarrollo
pip install -e ".[dev]"
```

---

## Conceptos clave

### Bloques con/sin estado y orden topológico

- Fase **A**: se evalúan **outputs** de todos los bloques (con y sin estado) en **orden topológico de feedthrough** para rellenar inputs.
- Fase **B**: solo los **bloques con estado** actualizan su estado según el solver explícito (Euler/RK…).
- Con **solvers globales** (p. ej., TorchDiffEq, Trapezoidal implícito), A/B se “embeben” dentro del solve monolítico.

### Solvers: globales vs no globales

- **No globales (por-bloque)**: el `Simulator` llama a `step_continuous(...)` por bloque con estado (Euler, RK4, RK45…).
- **Globales (monolíticos)**: un único `step_all(model, states, dt, t, externals_time_fn)` avanza **todo el modelo** (TorchDiffEq, y en F2 Trapezoidal).

---

## Externals: modo **estricto con alias** ✅

Al conectar una entrada externa:

```python
m.connect("Fext", "sum.Fext")                     # 'Fext' es el nombre de la señal externa
m.blocks["sum"].alias_inputs({"Fext":"in3", "Fs":"in1", "Fd":"in2"})
```

La conexión se normaliza al **puerto canónico** (`in3`). El inyector acepta:

- la **clave canónica**: `{"Fext": {"in3": tensor}}`
- o el **alias** declarado: `{"Fext": {"Fext": tensor}}`

Si llega otra clave (p. ej. `"wrong_port"`), se lanza:

```
KeyError: "External 'Fext': falta la clave 'in3'. Claves aceptadas para ese destino: ['in3','Fext'].
Claves disponibles: ['wrong_port']"
```

> Si dos destinos deben recibir valores **distintos**, usa **externals diferentes** (p. ej. `F1`, `F2`) y conéctalos por separado.

### Helpers de externals (port obligatorio)

Las *factories* en `kombox/externals.py` **siempre exigen `port`** (canónico o alias válido del bloque destino):

```python
from kombox.externals import make_external_step, make_external_constant, combine_externals

# MSD: el sumador tiene alias {"Fext":"in3"} -> puerto canónico "in3"
Fext = make_external_step("Fext", port="in3", t0=0.010, y_before=0.0, y_after=1.0,
                          batch_size=B, width=1, device=device, dtype=torch.float32)

# Motor DC: el bloque usa puerto canónico "tau_load" para el par de carga
tau_ext = make_external_step("tau", port="tau_load", t0=0.20, y_before=0.0, y_after=0.02,
                             batch_size=B, width=1, device=device, dtype=torch.float32)

ext_fn = combine_externals(Fext, tau_ext)
```

---

## API DAE mínima (MVP)

Pensada para ampliar a lazos algebraicos y restricciones sin romper autograd.

### En bloques

```python
class MyBlock(Block):
    # ...
    def algebraic_residual(self, t, state, inputs, params):
        # Por defecto: return {}
        # Si procede: devuelve dict o Tensor con g(...)=0
        return {}
```

### En el modelo

```python
# Restringe globalmente h(t, x, z, u, params) = 0
model.add_constraint_eq("holonomic", fn_h)

# Construye el residual global (se concatena lo local + eq. globales)
r = model.build_residual(t, x_all, u_all, z=None, params_all=None)
```

> Los solvers implícitos (p. ej. Trapezoidal + Newton-Krylov) consumirán `build_residual` para cerrar lazos. En v0.1.0, `TorchDiffEqSolver` resuelve ODE global.

---

## Uso de solvers

### No global (por-bloque)
```python
from kombox.core.simulator import Simulator
from kombox.core.solvers import RK45Solver

sim = Simulator(model, solver=RK45Solver())
sim.simulate(dt=1e-3, total_time=1.0, externals_fn=ext_fn, progress=True)  # progreso opcional
```

### Global (TorchDiffEq)
```python
from kombox.core.simulator import Simulator
from kombox.core.solvers import TorchDiffEqSolver

sim = Simulator(model, solver=TorchDiffEqSolver(method="dopri5", rtol=1e-6, atol=1e-8))
sim.simulate(dt=1e-3, steps=1000, externals_fn=ext_fn)
```

---

## Ejemplos

```bash
python kombox/examples/example_001_msd.py
python kombox/examples/example_009_dcmotor_pi.py
```

- MSD (sumador con alias `Fext → in3`):  
  `{"Fext": {"in3": tensor}}` (o `{"Fext": {"Fext": tensor}}` si usas alias).
- Motor DC (puerto `tau_load`):  
  `{"tau": {"tau_load": tensor}}`.

---

## Tests

```bash
pytest -q
```

- Si no tienes `torchdiffeq`, los tests relacionados se **saltan**.
- La suite incluye:
  - rutas solver **no global** y **global**,
  - **externals estrictos con alias** (OK y error),
  - checkpoint/restore de tiempo y estados,
  - **smoke DAE** (si añades `newton_krylov` y `trapezoidal`).

---

## Migración rápida

1) **Externals estrictos**: usa la **clave canónica** del puerto o su **alias** declarado.
2) **Helpers de externals**: ahora **siempre** requieren `port=` (canónico o alias).
3) **API DAE** (opt-in): `algebraic_residual`, `add_constraint_eq`, `build_residual`.

---

## Contribuir

- Estilo: tipado liviano (`typing`), **no** usar `.detach()` en rutas de cálculo (autograd-first).
- Añade tests; puedes ejecutar solo un archivo:
  ```bash
  pytest -q tests/test_externals_and_errors.py::test_external_injection_ok
  ```
- Abre un PR con una descripción clara de la API y casos de test.

---

## Roadmap (F2 breve)

- SCC + residual de lazos algebraicos en `build_residual`.
- Solver implícito **Trapezoidal** acoplado (x,z) + `Newton-Krylov` JVP-only.
- Inicialización consistente de DAEs.
- Estabilización (Baumgarte) y proyección pos-paso.
