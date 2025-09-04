# KomBox — Roadmap Detallado

Versión inicial orientada a ejecución por fases, con **criterios de aceptación**, **riesgos** y **métricas**. Este documento parte del estado actual del repo y de los puntos 3–5 de la guía.

---

## Fase 0 · Consolidación inmediata (1–2 semanas)
**Objetivo:** estabilizar la base actual y dejar checklist de calidad.

- **API freeze (v0.2.0):**
  - Congelar nombres/firmas públicas en `core/`, `blocks/`, `simulator`, `recorders`, `utils.externals`.
  - Añadir `@deprecated` y `CHANGELOG.md` para cualquier renombre.
- **CI y estilo:**
  - `ruff + black + isort + mypy` (modo relajado) + `pre-commit`.
  - Matriz CI: Python 3.9–3.12 (CPU; GPU opcional si runner).
- **Tests críticos:**
  - Determinismo con `torch.manual_seed`.
  - Paridad CPU/GPU en ejemplos básicos.
  - Checkpoints: test `preserves_time_and_progress`.
- **Criterio de aceptación:**
  - `pytest` verde.
  - Docs **P3–P5** enlazadas desde `README`.
  - Cobertura ≥ **80%** en `core/` y bloques críticos.

---

## Fase 1 · Eventos, DAEs y Solvers (3–4 semanas)
**Objetivo:** eventos “de verdad”, proyección de restricciones robusta y selección automática de solver.

- **Eventos (ZeroCrossing) dentro de bloques:**
  - API: `guard(...)` / `on_event(...)` (reset estado, jumps, cambio de params).
  - Detección de cruce con **interpolación Hermite** (para `dopri5`) y **bisección** (paso fijo).
  - Priorización y reintentos si hay cascada de eventos en el mismo `t`.
- **Restricciones y DAEs:**
  - `Model.add_constraint_eq/ineq`, inicialización consistente, **proyección pos‑paso** (Levenberg–Marquardt).
  - Métrica de violación `||g||` registrada en recorder.
- **Solvers:**
  - `auto_solver_for(model)`: sin ciclos → `Euler/RK4`; con ciclos algebraicos → `Trapezoidal` o `TorchDiffEq(dopri5)`.
  - `TorchDiffEqSolver(method="dopri5")` con `use_adjoint` opcional.
- **Criterio de aceptación:**
  - Suite con 4 casos reales: saturación, fricción **stick–slip**, contacto unilateral, relé.
  - Test de proyección mantiene `||g|| < tol` tras cada paso.

---

## Fase 2 · Batch Mode (las 3 opciones) (2–3 semanas)
**Objetivo:** soportar tres estrategias de batch según complejidad/heterogeneidad.

1) **Lock‑step (vectorizado puro)**  
   - Mismo `dt` y misma secuencia de pasos para todo el batch.  
   - **Uso:** sistemas regulares, sin eventos divergentes.  
   - **Aceptación:** speedup ~O(B) y equivalencia con bucle B=1 repetido.

2) **Lock‑step con máscaras (eventos divergentes)**  
   - Un único paso global; entradas/updates por muestra con **máscaras**.  
   - Agrupar por estado de evento para evitar ramificaciones explosivas.  
   - **Aceptación:** caso stick–slip donde una parte del batch entra en “stick” y otra no; resultados = per‑sample.

3) **Per‑sample / Bucketing (asincrónico controlado)**  
   - `t_next[i]` por muestra; **buckets** por `dt`/evento/topología para minimizar overhead.  
   - API: `BatchMode={"lockstep"|"masked"|"bucket"}` en `Simulator`.  
   - **Aceptación:** ≥ **30%** más rápido que per‑sample naíf en batch heterogéneo.

---

## Fase 3 · Librería de bloques (3–5 semanas)
**Objetivo:** cubrir un “MVP simscape‑like” y control clásico.

- **Mecánica 1D/2D:** `Mass1D/2D`, `Spring`, `Damper`, `RigidLink2D`, `PinJoint2D`, `Ground`.
- **Fricción:** Coulomb + viscosa con evento (stick–slip).
- **Eléctrica básica:** `Resistor`, `Capacitor`, `Inductor`, `SourceV/I`.
- **Control:** `Adder`, `Gain`, `Saturation`, `PID` (discreto/continuo).
- **Criterio de aceptación:** ejemplos integrados (Fase 5) reproducen resultados de referencia.

---

## Fase 4 · Rendimiento y estabilidad (2–3 semanas)
**Objetivo:** acelerar y estabilizar.

- `torch.compile` (modo dinámico) en ruta caliente del `Simulator`.
- Preasignación de buffers, fusión de kernels en recorders.
- **Mixed precision** (fp16/bf16) opcional.
- Benchmarks reproducibles: `bench/` con tamaños B = {1, 8, 64, 256}.
- **Criterio de aceptación:** ≥ **1.5×** vs baseline en lock‑step; sin fugas de memoria.

---

## Fase 5 · Ejemplos y documentación (2–3 semanas)
**Objetivo:** elevar UX y confianza del usuario.

- **Ejemplos guiados** (en `examples/` + notebooks):
  1. Masa–muelle–amortiguador + optimización de `k`.
  2. PID: seguimiento de referencia.
  3. Fricción estática/dinámica (stick–slip) con eventos.
  4. Contacto unilateral (inequidad + evento).
  5. **Ascenso de lanzador** (🚀).
- **Docs unificadas (mkdocs‑material):**
  - Integrar PDFs de P3–P5; páginas API con `pdoc`/`mkdocstrings`.
- **Criterio de aceptación:** `mkdocs build` sin *warnings*; enlaces CI automáticos.

---

## Fase 6 · GUI (Electron) — MVP (4–6 semanas)
**Objetivo:** un front minimal pero útil; “crear modelo, conectar, simular, ver”.

- **Arquitectura:** Electron + React (front) ↔ FastAPI (backend Python).
- **Pantallas:**
  - Inicio (proyecto, abrir/crear).
  - **Model Builder**: lista de bloques, editor de conexiones (grafo), inspector (parámetros/estados).
  - Simulación: dt/solver/batch mode, externos (editor de señales).
  - Visor: plots, export (NPZ/CSV).
- **Formato de proyecto:** `.kombox.json` (bloques, wiring, params, externos).
- **Criterio de aceptación:** recrear el ejemplo MSD sin escribir código; exportar a script reproducible.

---

## Fase 7 · Publicación y adopción (1–2 semanas)
**Objetivo:** empaquetar y hacer release.

- Packaging: `pyproject.toml`, ruedas manylinux, `pip install kombox`.
- Versionado **SemVer**: v0.3.0 (post‑GUI MVP) + `CONTRIBUTING.md`.
- Plantillas de **Issues/PRs** y `CODE_OF_CONDUCT.md`.
- **Criterio de aceptación:** instalación limpia en entorno fresco; ejemplo MSD < **60 s** post‑install.

---

## Pistas técnicas y decisiones
- **torchdiffeq** se mantiene como **opcional** (métodos adaptativos); `Trapezoidal` cubre ciclos sin dependencia externa.
- **Interpolación Hermite**: necesaria para **eventos** con `dopri5`; *root‑finding* por bisección/secante.
- **Topología y ciclos**: mantener `analyze_topology()` y SCCs para guiar `auto_solver_for`.
- **Recorders**: `MemoryRecorder` (RAM), `NPZChunkRecorder` (disco por partes). `ParquetRecorder` (opcional).

---

## Riesgos y mitigación
- **Divergencia por eventos simultáneos:** prioridad y límite de reintentos por `t`; registrar advertencia y continuar.
- **Heterogeneidad de batch:** `bucket` mode para minimizar iterar caso a caso.
- **Coste memoria en autograd:** adjoint para `dopri5`, o pérdidas por ventanas (*windowed loss*).

---

## Métricas de éxito
- Tiempo de simulación MSD (B=128) lock‑step: **≤ 40 ms/step** (CPU).
- `||g||` medio pos‑paso **< 1e−8** en casos con restricciones.
- 5 ejemplos completos reproducibles desde GUI y desde script.
- Cobertura **≥ 85%** en `core/` al finalizar Fase 5.

---

## Backlog (ordenado)
- [ ] Interpolación Hermite + *root‑finding* de eventos (bisección/secante).
- [ ] `BatchMode` (lockstep/masked/bucket) con benchmarks.
- [ ] Complementariedades (`add_complementarity`) y solver LM/VI (si aplica).
- [ ] Bloques 2D básicos (`PinJoint2D`, `RigidLink2D`, `Ground`).
- [ ] `ParquetRecorder` y utilidades de E/S.
- [ ] `mkdocs` + GitHub Pages.
- [ ] GUI: editor de señales externas (step/ramp/sine/square), drag‑and‑drop de wiring.

---

_Última edición: generado automáticamente a partir de la conversación y del estado del repo._
