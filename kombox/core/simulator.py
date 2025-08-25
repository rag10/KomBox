# kombox/core/simulator.py
# from __future__ import annotations
from typing import Dict, Mapping, Optional, Callable

import torch

import math
import time

from .model import Model
from .block import Block
from .solvers import SolverBase, EulerSolver


class Simulator:
    """
    Runtime en dos fases. Ahora lee/escribe estados directamente en `model`.
    """

    def __init__(
        self,
        model: Model,
        *,
        solver: Optional[SolverBase] = None,
        validate_io: bool = False,
        strict_numerics: bool = False,
    ):
        if not model._built:
            raise RuntimeError("Simulator: Model.build() debe llamarse antes.")
        if not model.has_states:
            raise RuntimeError("Simulator: el Model no tiene estados. Llama a model.initialize(...).")

        self.model = model
        self.solver = solver if solver is not None else EulerSolver()
        self._last_dt: float | None = None
        self.validate_io = bool(validate_io)
        self.strict_numerics = bool(strict_numerics)

        any_state = next(iter(self.model.states.values()))
        if any_state.ndim != 2:
            raise ValueError(f"Simulator: cada estado debe ser 2D (B,S), recibido {tuple(any_state.shape)}.")

        self._B = int(any_state.shape[0])
        self._device = any_state.device
        self._dtype = any_state.dtype

        self.t: float = 0.0
        self.k: int = 0

        # buffers por bloque
        self._inbuf: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in self.model.blocks.keys()}
        self._outs_cache: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in self.model.blocks.keys()}

        # plan del modelo
        self._downstream = self.model._downstream
        self._orderA = self.model._order_outputs
        self._ext_in = self.model._ext_in
        self._ext_out = self.model._ext_out

    # ---------- atajos ----------
    @property
    def B(self) -> int: return self._B
    @property
    def device(self) -> torch.device: return self._device
    @property
    def dtype(self) -> torch.dtype: return self._dtype
    @property
    def states(self) -> Dict[str, torch.Tensor]: return self.model.states

    def reset_time(self): self.t = 0.0; self.k = 0

    # ---------- helpers ----------
    @staticmethod
    def _merge_two_level(
        base: Optional[Mapping[str, Mapping[str, torch.Tensor]]],
        over: Optional[Mapping[str, Mapping[str, torch.Tensor]]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        if base:
            for n, d in base.items(): out[n] = dict(d)
        if over:
            for n, d in over.items():
                out.setdefault(n, {}); out[n].update(d)
        return out

    def _inject_externals(self, eff_ext: Mapping[str, Mapping[str, torch.Tensor]]):
        # Limpia buffers
        for d in self._inbuf.values():
            d.clear()

        for ext_name, targets in self._ext_in.items():
            if ext_name not in eff_ext:
                raise KeyError(f"Falta entrada externa '{ext_name}'. Esperadas: {self.model.required_externals()}")

            src_dict = eff_ext[ext_name]

            for (blk_name, port_can) in targets:
                blk = self.model.blocks[blk_name]

                # 1) ¿viene con el nombre canónico?
                if port_can in src_dict:
                    ten = src_dict[port_can]
                else:
                    # 2) ¿viene con alguno de los alias que apuntan a ese canónico?
                    alias_for_port = [a for (a, c) in blk._in_alias.items() if c == port_can]
                    ten = None
                    for a in alias_for_port:
                        if a in src_dict:
                            ten = src_dict[a]
                            break

                    # 3) Estricto: si no hay ni canónico ni alias → error claro
                    if ten is None:
                        accepted = [port_can] + alias_for_port
                        raise KeyError(
                            f"External '{ext_name}': falta la clave '{port_can}'. "
                            f"Claves aceptadas para ese destino: {accepted}. "
                            f"Claves disponibles: {sorted(list(src_dict.keys()))}"
                        )

                self._inbuf[blk_name][port_can] = ten

    # ---------- un paso ----------
    def step(
        self,
        dt: float,
        *,
        externals: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
        externals_fn: Optional[Callable[[float, int], Mapping[str, Mapping[str, torch.Tensor]]]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        ext_from_fn = externals_fn(self.t, self.k) if externals_fn is not None else None
        eff_ext = self._merge_two_level(externals, ext_from_fn)

        # --- Ruta solver GLOBAL (torchdiffeq) ---
        if getattr(self.solver, "is_global", False):
            # construir función de externals dependiente de t para los subpasos internos
            def ext_time_fn(tnow: float):
                # mezcla externals "fijos" (eff_ext) con externals_fn(tnow, k_est)
                base = eff_ext or {}
                over = None
                if externals_fn is not None:
                    # k estimado a partir de t y último dt (no crítico; mayoría de externals solo dependen de t)
                    k_est = int(round((tnow - (self.t - (self._last_dt or 0.0))) / (self._last_dt or 1.0))) if self._last_dt else self.k
                    over = externals_fn(tnow, k_est)
                return self._merge_two_level(base, over)

            self._last_dt = float(dt)
            new_states, outs_end = self.solver.step_all(self.model, self.states, float(dt), self.t,
                                                        externals_time_fn=ext_time_fn)

            # Validaciones y escritura de estados
            if self.validate_io:
                for bname, blk in self.model.blocks.items():
                    blk._validate_outputs(outs_end[bname], batch_size=self.B)
            if self.strict_numerics:
                for bname, st in new_states.items():
                    if st.numel()>0 and not torch.isfinite(st).all():
                        raise FloatingPointError(f"{bname}.state: NaN/Inf.")
                for bname, outs in outs_end.items():
                    for pname, ten in outs.items():
                        if not torch.isfinite(ten).all():
                            raise FloatingPointError(f"{bname}.{pname}: NaN/Inf.")

            # aplicar
            self.model.set_states(new_states, validate_shapes=False)
            self._outs_cache = outs_end
            self.k += 1; self.t += float(dt)
            return self._outs_cache

        # --- Ruta clásica por-bloque (Euler/RK4/RK45) ---
        self._inject_externals(eff_ext)

        B = self._B

        # FASE A: outs
        for bname in self._orderA:
            blk: Block = self.model.blocks[bname]
            ins = self._inbuf[bname]
            outs = blk._expose_outputs(self.states[bname], ins, self.t)
            self._outs_cache[bname] = outs
            for (port_name, ten) in outs.items():
                for (dst_blk, dst_port) in self._downstream.get((bname, port_name), []):
                    self._inbuf[dst_blk][dst_port] = ten

        # FASE B: advance
        new_states: Dict[str, torch.Tensor] = {}
        for bname, blk in self.model.blocks.items():
            ins = self._inbuf[bname]
            if self.validate_io: blk._validate_inputs(ins, batch_size=B)
            new_st, outs2 = blk._advance(self.states[bname], ins, float(dt), self.t, self.solver)
            if self.validate_io: blk._validate_outputs(outs2, batch_size=B)
            if self.strict_numerics:
                if new_st.numel() > 0 and not torch.isfinite(new_st).all():
                    raise FloatingPointError(f"{bname}.state: NaN/Inf.")
                for pname, ten in outs2.items():
                    if not torch.isfinite(ten).all():
                        raise FloatingPointError(f"{bname}.{pname}: NaN/Inf.")
            new_states[bname] = new_st
            self._outs_cache[bname] = outs2

        # Actualiza estados en el propio modelo
        self.model.set_states(new_states, validate_shapes=False)

        self.k += 1
        self.t += float(dt)
        return self._outs_cache

    # ---------- bucle ----------
    def simulate(
        self,
        *,
        dt: float,
        total_time: Optional[float] = None,
        steps: Optional[int] = None,
        externals: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
        externals_fn: Optional[Callable[[float, int], Mapping[str, Mapping[str, torch.Tensor]]]] = None,
        recorder: Optional[object] = None,
        reset_time: bool = True,
        # progreso (opcional)
        progress: bool = False,
        progress_interval: float = 1.0,
        progress_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ):
        """
        Ejecuta un bucle de simulación. Si 'progress=True', imprime (o envía a 'progress_fn')
        una línea de estado cada ~progress_interval segundos de reloj.
        """
        if (total_time is None) == (steps is None):
            raise ValueError("Especifica exactamente uno: steps o total_time.")
        if steps is None:
            steps = int(round(float(total_time) / float(dt)))
        if reset_time:
            self.reset_time()

        last: Dict[str, Dict[str, torch.Tensor]] = {}
        rec_has_time = False
        if recorder is not None and hasattr(recorder, "on_step"):
            try:
                rec_has_time = recorder.on_step.__code__.co_argcount >= 4
            except Exception:
                rec_has_time = False

        # ----- Config progreso -----
        start_wall = time.monotonic()
        last_report = start_wall
        start_t = self.t
        start_k = self.k
        target_steps = int(steps)

        # ----- Bucle principal -----
        for _ in range(int(steps)):
            last = self.step(dt, externals=externals, externals_fn=externals_fn)

            if recorder is not None and hasattr(recorder, "on_step"):
                if rec_has_time:
                    recorder.on_step(self.t, last, self.states)
                else:
                    recorder.on_step(last, self.states)

            # Progreso (acotado por tiempo)
            if progress:
                now = time.monotonic()
                if now - last_report >= float(progress_interval):
                    wall_elapsed = now - start_wall
                    done_steps = self.k - start_k
                    sim_elapsed = self.t - start_t

                    # Métricas
                    steps_per_s = (done_steps / wall_elapsed) if wall_elapsed > 0 else float("inf")
                    speed_x = (sim_elapsed / wall_elapsed) if wall_elapsed > 0 else float("inf")  # seg simulados / seg real
                    frac = (done_steps / target_steps) if target_steps > 0 else 0.0
                    remaining_steps = max(0, target_steps - done_steps)
                    eta_s = (remaining_steps / steps_per_s) if steps_per_s > 0 else float("nan")

                    info = {
                        "t": float(self.t),
                        "k": int(self.k),
                        "dt": float(dt),
                        "elapsed_wall_s": float(wall_elapsed),
                        "speed_steps_per_s": float(steps_per_s),
                        "speed_x": float(speed_x),
                        "progress": float(frac),
                        "eta_s": float(eta_s),
                        "target_steps": int(target_steps),
                    }

                    if progress_fn:
                        progress_fn(info)
                    else:
                        print(
                            f"[KomBox] {done_steps}/{target_steps} ({frac*100:5.1f}%)  |  "
                            f"t={self.t:.3f}s  |  Elapsed={wall_elapsed:5.1f}s  |  Speed={speed_x:5.1f}x  |  "
                            f"Remaining ≈ {eta_s:5.1f}s",
                            flush=True,
                        )
                    last_report = now

        return last
