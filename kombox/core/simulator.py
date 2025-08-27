# kombox/core/simulator.py
# from __future__ import annotations
from typing import Dict, Mapping, Optional, Callable, Tuple, List

import torch

import math
import time

from .model import Model
from .block import Block
from .solvers import SolverBase, EulerSolver, auto_solver_for


def _flatten_states(states: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str, slice, Tuple[int,int]]]]:
    """Concatena todos los estados (B,S_i) por columnas → (B, S_total). Devuelve tensor y metainfo."""
    if len(states) == 0:
        return torch.empty((1,0), dtype=torch.float32), []
    B = None
    metas: List[Tuple[str, slice, Tuple[int,int]]] = []
    cols = []
    col_start = 0
    for name, X in states.items():
        if X is None:
            continue
        if B is None:
            B = X.shape[0]
        S = X.shape[1] if X.ndim == 2 else 0
        if S == 0:
            continue
        cols.append(X)
        metas.append((name, slice(col_start, col_start + S), (B, S)))
        col_start += S
    if len(cols) == 0:
        return torch.empty((B if B is not None else 1, 0), dtype=list(states.values())[0].dtype if states else torch.float32, device=list(states.values())[0].device if states else torch.device("cpu")), metas
    Xall = torch.cat(cols, dim=1)
    return Xall, metas

def _unflatten_states(Xall: torch.Tensor, metas: List[Tuple[str, slice, Tuple[int,int]]], proto: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reconstruye el dict de estados a partir de Xall y las metas; conserva device/dtype por tensor."""
    out: Dict[str, torch.Tensor] = {}
    for name in proto.keys():
        out[name] = proto[name]  # por defecto, mantener referencias
    for name, slc, (B, S) in metas:
        out[name] = Xall[:, slc]
    return out

def _collect_global_constraints(model, t: float, states: Dict[str, torch.Tensor], inbuf: Dict[str, Dict[str, torch.Tensor]], z: torch.Tensor = None) -> torch.Tensor:
    """Devuelve solo las restricciones globales g(x,...) concatenadas (B, Rg). Si no hay, retorna (B,0)."""
    r, detail = model.build_residual(t, states=states, inbuf=inbuf, z=z)
    # filtra claves globales: usan el prefijo "__global__::"
    pieces = []
    B = r.shape[0] if r.ndim == 2 else 1
    for k, v in detail.items():
        if isinstance(k, str) and k.startswith("__global__::"):
            vv = v if v.ndim == 2 else v.view(B, -1)
            pieces.append(vv)
    if len(pieces) == 0:
        return torch.zeros((B, 0), dtype=r.dtype, device=r.device)
    return torch.cat(pieces, dim=1)

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
        self.model = model
        self.solver = solver if solver is not None else auto_solver_for(model)
        self._last_dt: float | None = None
        self.validate_io = bool(validate_io)
        self.strict_numerics = bool(strict_numerics)

        if model.has_states:
            any_state = next(iter(self.model.states.values()))
            if any_state.ndim != 2:
                raise ValueError(f"Simulator: cada estado debe ser 2D (B,S), recibido {tuple(any_state.shape)}.")
            self._B = int(any_state.shape[0])
            self._device = any_state.device
            self._dtype = any_state.dtype
        else:
            # Permite flujos puramente algebraicos (initialize_consistent) sin initialize()
            self._B = 0
            self._device = torch.device("cpu")
            self._dtype = torch.float32

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

        # proyección pos-paso para restricciones globales
        self._proj_enabled = False
        self._proj_tol = 1e-8
        self._proj_maxit = 5
        self._proj_damping = 1e-8

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

    # --- inicialización consistente de DAEs ---------------------
    def initialize_consistent(
        self,
        t0: float = 0.0,
        externals_fn=None,
        solver=None,
        tol: float = 1e-8,
        max_iter: int = 20,
    ):
        """
        Inicialización consistente con soporte de z global:
        - Construye inbuf en t0 con externals.
        - Ensambla residual global R(t0, states=None, inbuf, z).
        - Si hay z (model._z_dim > 0), resuelve F(z)=0 con solver (NewtonKrylov por defecto).
        No modifica estados. Devuelve info diagnóstica.
        """
        model = self.model

        # 1) Externals efectivos en t0
        eff_ext = {}
        if externals_fn is not None:
            out = externals_fn(float(t0), 0)
            if not isinstance(out, dict):
                raise TypeError("externals_fn debe devolver un dict {ext_name: {port: Tensor}}")
            eff_ext = out

        # 2) inbuf temporal a partir del wiring de externals
        inbuf = {n: {} for n in model.blocks.keys()}
        if hasattr(self, "_ext_in") and isinstance(self._ext_in, dict) and len(self._ext_in) > 0:
            for ext_name, targets in self._ext_in.items():
                if ext_name not in eff_ext:
                    raise KeyError(f"Falta entrada externa '{ext_name}'. Esperadas: {model.required_externals()}")
                src_dict = eff_ext[ext_name]
                for (blk_name, port_can) in targets:
                    if port_can not in src_dict:
                        raise KeyError(
                            f"External '{ext_name}': falta la clave '{port_can}'. "
                            f"Claves disponibles: {sorted(list(src_dict.keys()))}"
                        )
                    inbuf[blk_name][port_can] = src_dict[port_can]

        # Compatibilidad: permitir que constraints lean externals por grupo si no hay wiring
        for grp, ports in eff_ext.items():
            if grp not in model.blocks:
                inbuf[grp] = dict(ports)

        # 3) Inferir B/device/dtype desde inbuf (sin depender del residual)
        B, device, dtype = None, torch.device("cpu"), torch.float32
        found_t = None
        for bd in inbuf.values():
            for ten in bd.values():
                if isinstance(ten, torch.Tensor) and ten.numel() > 0:
                    found_t = ten
                    break
            if found_t is not None:
                break
        if found_t is not None:
            B = int(found_t.shape[0])
            device = found_t.device
            dtype = found_t.dtype
        else:
            B = 1  # por defecto si no hay tensores en inbuf

        # 4) Si hay z declarado, crear z0 de ceros para el residual inicial
        z_dim = int(getattr(model, "_z_dim", 0) or 0)
        z0 = None
        if z_dim > 0:
            z0 = torch.zeros((B, z_dim), dtype=dtype, device=device)

        # 5) Residual inicial (con z0 si procede)
        residual0, detail0 = model.build_residual(t0, states=None, inbuf=inbuf, z=z0)
        if residual0.ndim == 1:
            residual0 = residual0.view(1, -1)
        rn0 = torch.sqrt(torch.sum(residual0 * residual0, dim=1) + 1e-16)  # (B,)

        # 6) Resolver Z si procede
        did_solve = False
        z_star = None
        if z_dim > 0:
            if solver is None:
                from ..core.algebraic.newton_krylov import NewtonKrylov
                solver = NewtonKrylov(mode="jfnk", max_iter=max_iter, tol=tol)

            def Fz(z):
                r = model.build_residual(t0, states=None, inbuf=inbuf, z=z)
                if isinstance(r, tuple):
                    r = r[0]
                return r

            z_star = solver.solve(Fz, z0)  # parte de z0 (ceros)
            did_solve = True

            residual1, detail1 = model.build_residual(t0, states=None, inbuf=inbuf, z=z_star)
            if residual1.ndim == 1:
                residual1 = residual1.view(1, -1)
            rn1 = torch.sqrt(torch.sum(residual1 * residual1, dim=1) + 1e-16)
            return {
                "residual": residual1,
                "detail": detail1,
                "residual_norm": float(rn1.max().detach().cpu()),
                "residual_norm_before": float(rn0.max().detach().cpu()),
                "B": B,
                "z_dim": z_dim,
                "z_star": z_star.detach(),
                "did_solve": did_solve,
                "t0": float(t0),
            }

        # 7) Sin z: devolvemos residual inicial
        return {
            "residual": residual0,
            "detail": detail0,
            "residual_norm": float(rn0.max().detach().cpu()),
            "B": B,
            "z_dim": 0,
            "z_star": None,
            "did_solve": False,
            "t0": float(t0),
        }
    
    def enable_constraint_projection(self, enabled: bool = True, tol: float = 1e-8, max_iter: int = 5, damping: float = 1e-8):
        """Activa/desactiva proyección pos-paso para g(x)=0."""
        self._proj_enabled = bool(enabled)
        self._proj_tol = float(tol)
        self._proj_maxit = int(max_iter)
        self._proj_damping = float(damping)
        return self

    def _project_states_once(self, states: Dict[str, torch.Tensor], inbuf: Dict[str, Dict[str, torch.Tensor]], t: float) -> Dict[str, torch.Tensor]:
        """Un paso LM: resuelve min||δx|| s.t. g(x+δx)=0 ≈ línea: (Jg) δx = -g; dx = argmin || (Jg) dx + g ||_2 + λ||dx||."""
        # 1) Aplanar
        Xall, metas = _flatten_states(states)
        if Xall.shape[1] == 0:
            return states  # nada que proyectar si no hay estados
        Xall = Xall.clone().requires_grad_(True)

        # 2) Define g(X) como solo restricciones globales
        def g_only(Xflat: torch.Tensor) -> torch.Tensor:
            st = _unflatten_states(Xflat, metas, states)
            g = _collect_global_constraints(self.model, t, st, inbuf, z=None)  # z no participa aquí
            # g: (B, Rg)
            return g

        # 3) Evalúa g y Jacobiano por batch
        g = g_only(Xall)  # (B,Rg)
        B = g.shape[0]
        Rg = g.shape[1] if g.ndim == 2 else 0
        if Rg == 0:
            return states  # no hay restricciones globales

        # --- NUEVO: jacobiano batcheado sin in-place ---
        # J tiene forma (B, Rg, B, S). Para cada batch b, nos quedamos con J[b, :, b, :].
        J_full = torch.autograd.functional.jacobian(g_only, Xall, create_graph=True)  # (B,Rg,B,S)
        # Seguridad: algunas versiones devuelven (Rg,B,S,...) — aquí asumimos (B,Rg,B,S).
        assert J_full.ndim == 4, f"jacobian shape inesperada: {tuple(J_full.shape)}"

        # 4) Resuelve normales con damping (Levenberg)
        S = Xall.shape[1]
        eyeS = torch.eye(S, device=Xall.device, dtype=Xall.dtype)
        Xnew = []
        for b in range(B):
            gb = g[b].reshape(-1, 1)                 # (Rg,1)
            Jb = J_full[b, :, b, :]                  # (Rg,S)
            AtA = Jb.T @ Jb + self._proj_damping * eyeS  # (S,S)
            rhs = - Jb.T @ gb                        # (S,1)
            dx = torch.linalg.solve(AtA, rhs).view(1, S)  # (1,S)
            Xnew.append(Xall[b:b+1] + dx)

        Xupd = torch.cat(Xnew, dim=0)  # (B,S)

        # 5) Reconstruir dict
        st_new = _unflatten_states(Xupd, metas, states)
        return st_new

    def _project_states(self, states: Dict[str, torch.Tensor], inbuf: Dict[str, Dict[str, torch.Tensor]], t: float) -> Dict[str, torch.Tensor]:
        """Itera _project_states_once hasta tolerancia o max_iter."""
        if not getattr(self, "_proj_enabled", False):
            return states
        st = states
        for _ in range(getattr(self, "_proj_maxit", 5)):
            g = _collect_global_constraints(self.model, t, st, inbuf, z=None)
            if g.shape[1] == 0:
                break
            rn = torch.sqrt(torch.sum(g * g, dim=1) + 1e-16).max()
            if float(rn) < getattr(self, "_proj_tol", 1e-8):
                break
            st = self._project_states_once(st, inbuf, t)
        return st

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
        t0 = self.t
        
        if not self.model.has_states:
            raise RuntimeError(
                "Simulator.step requiere que el modelo tenga estados inicializados. "
                "Usa model.initialize(...) antes. Para DAEs puramente algebraicas, usa initialize_consistent()."
            )

        # si el modelo tiene lazos algebraicos y el solver no es global/implícito, abortar aquí
        if getattr(self.model, "_has_pure_algebraic_cycle", False):
            # por-bloque/explicitos suelen anunciarse con is_global = False
            if not getattr(self.solver, "is_global", False):
                cycles = getattr(self.model, "_algebraic_cycles", [])
                msg = " ; ".join(" -> ".join(c) for c in cycles) if cycles else "algebraic loop detected"
                raise ValueError(
                    f"Modelo contiene lazos algebraicos puros ({msg}). "
                    f"Usa un solver implícito/global (p.ej., Trapezoidal/DAE)."
                )

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
            # Proyección pos-paso (si hay restricciones globales y está activada)
            new_states = self._project_states(new_states, self._inbuf, t0 + dt)

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

        # Proyección pos-paso también para la ruta no-global
        new_states = self._project_states(new_states, self._inbuf, t0 + dt)
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
        if not self.model.has_states:
            raise RuntimeError(
                "Simulator.simulate requiere estados inicializados. "
                "Usa model.initialize(...). Para inicialización algebraica, usa initialize_consistent()."
            )

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

    # --- CheckPoints ---------------------------------------------
    def make_checkpoint(self, detach: bool = True) -> dict:
        """
        Captura un checkpoint del estado de la simulación.
        - detach=True: clona y *detachea* (no guarda la historia de autograd).
        - detach=False: clona manteniendo la historia (útil si vas a backpropagar).
        """
        # Estados del modelo en el dispositivo actual
        states_snap = {}
        for blk_name, ten in self.model.states.items():
            if detach:
                states_snap[blk_name] = ten.detach().clone()
            else:
                # clone() mantiene la conexión al grafo de autograd
                states_snap[blk_name] = ten.clone()

        chk = {
            "t": float(self.t),
            "k": int(self.k),
            "states": states_snap,
            # Si alguna vez guardas algo más (semillas, solver interno, etc.), añádelo aquí.
        }
        return chk

    def restore_checkpoint(self, chk: dict) -> None:
        """
        Restaura un checkpoint creado con make_checkpoint.
        No toca dtype/device: usa tal cual los tensores guardados en el checkpoint.
        """
        if not isinstance(chk, dict):
            raise TypeError("Checkpoint inválido: se esperaba un dict.")

        if "t" not in chk or "k" not in chk or "states" not in chk:
            raise KeyError("Checkpoint incompleto: faltan 't', 'k' o 'states'.")

        self.t = float(chk["t"])
        self.k = int(chk["k"])

        # Reemplaza completamente los estados del modelo
        states_in = chk["states"]
        for blk_name, ten in states_in.items():
            if blk_name not in self.model.states:
                raise KeyError(f"Checkpoint contiene bloque desconocido '{blk_name}'.")
            self.model.states[blk_name] = ten

        # Limpia buffers de entrada/auxiliares si existen
        if hasattr(self, "_inbuf") and isinstance(self._inbuf, dict):
            for d in self._inbuf.values():
                if isinstance(d, dict):
                    d.clear()
