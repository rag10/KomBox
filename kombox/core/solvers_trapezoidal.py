# kombox/core/solvers_trapezoidal.py
from __future__ import annotations
from typing import Dict, Callable, Optional, Tuple, List, Sequence, Literal
import inspect
import torch
from .solvers import SolverBase
from .algebraic.newton_krylov import NewtonKrylov  # import correcto

_ALLOWED_MODES = {"kkt", "kkt_baumgarte", "kkt_mixed"}

class TrapezoidalSolver(SolverBase):
    """
    Stepper implícito tipo Trapecio (global) con manejo de restricciones:
      - 'kkt':           todas con λ (KKT puro).
      - 'kkt_baumgarte': todas con estabilización de Baumgarte (sin λ).
      - 'kkt_mixed':     subconjunto con λ (lam_mask=True) y resto con Baumgarte (lam_mask=False).

    Además:
      - Re-eval de externals a t y t+dt.
      - Inyección de fuerzas por hooks tanto de KKT (λ) como de Baumgarte (λ_bg).
      - Validaciones claras de mask, hooks y shapes.
    """
    name = "trapezoidal"
    is_global: bool = True

    def __init__(
        self,
        algebraic_solver: Optional[NewtonKrylov] = None,
        *,
        constraint_mode: Literal["kkt","kkt_baumgarte","kkt_mixed"] = "kkt",
        lam_mask: Optional[Sequence[bool]] = None,
        baumgarte_alpha: float = 1.0,
        baumgarte_beta: float = 10.0,
        tol: float = 1e-8,
        max_iter: int = 20,
        debug: bool = False,
        # ---- compatibilidad hacia atrás ----
        baumgarte_enabled: Optional[bool] = None,
    ):
        super().__init__()
        self.alg = algebraic_solver or NewtonKrylov()
        self.is_global = True

        # Compat: mapear baumgarte_enabled -> constraint_mode si aplica
        cm = str(constraint_mode)
        if baumgarte_enabled is not None:
            if baumgarte_enabled and cm == "kkt":
                cm = "kkt_baumgarte"
            if baumgarte_enabled is False and constraint_mode == "kkt_baumgarte":
                raise ValueError("Inconsistencia: baumgarte_enabled=False pero constraint_mode='kkt_baumgarte'.")

        if cm not in _ALLOWED_MODES:
            raise ValueError(f"constraint_mode inválido: {cm!r}. Debe ser uno de {_ALLOWED_MODES}.")
        self.constraint_mode = cm

        self._lam_mask_in = list(lam_mask) if lam_mask is not None else None
        self.baumgarte_alpha = float(baumgarte_alpha)
        self.baumgarte_beta  = float(baumgarte_beta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.debug = bool(debug)

    # ----------------- helpers privados -----------------
    def _phaseA_eval(
        self, model, states, tnow: float,
        externals_time_fn: Optional[Callable[[float], Dict[str, Dict[str, torch.Tensor]]]]
    ) -> Tuple[Dict[str,Dict[str,torch.Tensor]], Dict[str,Dict[str,torch.Tensor]]]:
        """Evalúa fase A: outs y wiring → inbuf. Requiere externals_time_fn(t) si hay externals."""
        inbuf: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in model.blocks.keys()}
        ext_map = externals_time_fn(tnow) or {} if externals_time_fn is not None else {}
        # inyección estricta de externals solo si existen
        for ext_name, targets in model._ext_in.items():
            if ext_name not in ext_map:
                raise KeyError(f"Falta entrada externa '{ext_name}' en externals_fn(t={tnow:.6f}).")
            src_dict = ext_map[ext_name]
            for (blk, port) in targets:
                if port in src_dict:
                    inbuf[blk][port] = src_dict[port]
                elif len(src_dict) == 1:
                    inbuf[blk][port] = next(iter(src_dict.values()))
                else:
                    raise KeyError(f"External '{ext_name}' no provee puerto '{port}'.")
        outs: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in model.blocks.keys()}
        for bname in model._order_outputs:
            blk = model.blocks[bname]; st = states[bname]; ins = inbuf[bname]
            o = blk._expose_outputs(st, ins, tnow)
            outs[bname] = o
            for (pname, ten) in o.items():
                for (dst_blk, dst_port) in model._downstream.get((bname, pname), []):
                    inbuf[dst_blk][dst_port] = ten
        return inbuf, outs

    def _split_constraints_by_mask(self, model, lam_mask: Sequence[bool]) -> Tuple[List[int], List[int], List[str], List[str]]:
        """Devuelve (idx_lam, idx_bg, names_lam, names_bg) respetando el orden de registro."""
        names = [n for (n, _) in model._constraints]
        if len(lam_mask) != len(names):
            raise ValueError(
                f"lam_mask longitud={len(lam_mask)} != nº restricciones={len(names)}. "
                f"Restricciones: {names}"
            )
        idx_lam = [i for i, m in enumerate(lam_mask) if bool(m)]
        idx_bg  = [i for i, m in enumerate(lam_mask) if not bool(m)]
        names_lam = [names[i] for i in idx_lam]
        names_bg  = [names[i] for i in idx_bg]
        return idx_lam, idx_bg, names_lam, names_bg

    def _eval_constraints_list(self, model, t, states, inbuf) -> List[torch.Tensor]:
        """Evalúa g_i por restricción (lista) en orden de registro."""
        vals: List[torch.Tensor] = []
        for (name, fn) in model._constraints:
            try:
                sig = inspect.signature(fn); use_z = (len(sig.parameters) >= 5)
            except Exception:
                use_z = False
            g = fn(t, states, inbuf, model, None) if use_z else fn(t, states, inbuf, model)
            g = g if g.ndim == 2 else g.view(g.shape[0], -1)
            vals.append(g)
        return vals

    # ----------------- paso global -----------------
    def step_all(
        self,
        model,
        states: Dict[str, torch.Tensor],
        dt: float,
        t: float,
        externals_time_fn: Optional[Callable[[float], Dict[str, Dict[str, torch.Tensor]]]] = None
    ):
        from .block import ContinuousBlock, DiscreteBlock

        # --- Fase A en t ---
        inbuf_t0, _ = self._phaseA_eval(model, states, t, externals_time_fn)

        # --------- CASO SIN RESTRICCIONES: Trapecio puro y retorno seguro ---------
        if len(getattr(model, "_constraints", [])) == 0:
            # predictor con inbuf(t)
            pred_states: Dict[str, torch.Tensor] = {}
            for bname, blk in model.blocks.items():
                st0 = states[bname]
                if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                    dx0, _ = blk.ode(st0, inbuf_t0[bname], t)
                    pred_states[bname] = st0 + dt * dx0
                elif isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                    st1p, _ = blk.update(st0, inbuf_t0[bname], float(dt), float(t+dt))
                    pred_states[bname] = st1p
                else:
                    pred_states[bname] = st0
            # fase A en t+dt con predictor
            inbuf_t1, _ = self._phaseA_eval(model, pred_states, t + dt, externals_time_fn)
            # corrector trapecial (continuos) + actualización discretos
            new_states: Dict[str, torch.Tensor] = {k: v for k, v in states.items()}
            for bname, blk in model.blocks.items():
                st0 = states[bname]
                if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                    dx0, _ = blk.ode(st0, inbuf_t0[bname], t)
                    x_star = st0 + dt * dx0
                    dx1, _ = blk.ode(x_star, inbuf_t1[bname], t + dt)
                    st1 = st0 + 0.5 * dt * (dx0 + dx1)
                    new_states[bname] = st1
                elif isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                    st1, _ = blk.update(st0, inbuf_t1[bname], float(dt), float(t+dt))
                    new_states[bname] = st1
            _, outs_end = self._phaseA_eval(model, new_states, t + dt, externals_time_fn)
            return new_states, outs_end

        # --------- A partir de aquí: hay restricciones globales ---------
        if self.constraint_mode == "kkt_baumgarte" and self._lam_mask_in is not None:
            if any(self._lam_mask_in):
                raise ValueError("lam_mask no aplica en 'kkt_baumgarte'. Elimina o pasa todo False.")

        B = next(iter(states.values())).shape[0] if states else 1
        device = next(iter(states.values())).device if states else torch.device("cpu")
        dtype  = next(iter(states.values())).dtype  if states else torch.float32

        # tamaños por restricción y offsets
        sizes: List[int] = []
        for name, fn in model._constraints:
            try:
                sig = inspect.signature(fn); use_z = (len(sig.parameters) >= 5)
            except Exception:
                use_z = False
            g0_i = fn(t, states, inbuf_t0, model, None) if use_z else fn(t, states, inbuf_t0, model)
            g0_i = g0_i if g0_i.ndim == 2 else g0_i.view(g0_i.shape[0], -1)
            sizes.append(int(g0_i.shape[1]))
        ofs = [0]; [ofs.append(ofs[-1] + s) for s in sizes]
        Rg_total = int(ofs[-1])

        # máscara por modo
        if self.constraint_mode == "kkt_mixed":
            if self._lam_mask_in is None:
                raise ValueError("kkt_mixed requiere lam_mask (lista/tupla de bool).")
            if not isinstance(self._lam_mask_in, (list, tuple)) or not all(isinstance(b, (bool,)) for b in self._lam_mask_in):
                raise TypeError("lam_mask debe ser lista/tupla de bool.")
            idx_lam = [i for i, m in enumerate(self._lam_mask_in) if m]
            idx_bg  = [i for i, m in enumerate(self._lam_mask_in) if not m]
        elif self.constraint_mode == "kkt":
            idx_lam = list(range(len(sizes))); idx_bg = []
        else:  # kkt_baumgarte
            idx_lam = []; idx_bg = list(range(len(sizes)))

        # validación hooks en KKT
        for i in idx_lam:
            name_i = model._constraints[i][0]
            hooks = model._constraint_force_hooks.get(name_i, [])
            if len(hooks) == 0 and name_i not in model._constraint_forces:
                raise RuntimeError(
                    f"Restricción '{name_i}' marcada para λ no tiene force hooks. "
                    f"Registre con model.add_constraint_force('{name_i}', hook)."
                )

        # empaquetado λ por restricción a vector full
        def pack_full(lam_list: List[Optional[torch.Tensor]]) -> torch.Tensor:
            zfull = torch.zeros((B, Rg_total), device=device, dtype=dtype)
            for k, lam_k in enumerate(lam_list):
                if lam_k is None: 
                    continue
                s = sizes[k]
                if lam_k.shape != (B, s):
                    raise ValueError(f"λ para restricción {k} con shape {tuple(lam_k.shape)}; se esperaba {(B,s)}.")
                zfull[:, ofs[k]:ofs[k]+s] = lam_k
            return zfull

        # z compacto (solo λ-KKT)
        R_lam = sum(sizes[i] for i in idx_lam)
        z0_compact = torch.zeros((B, R_lam), device=device, dtype=dtype)

        def expand_to_full(zz_compact: torch.Tensor) -> torch.Tensor:
            if zz_compact is None or zz_compact.numel() == 0:
                return torch.zeros((B, Rg_total), device=device, dtype=dtype)
            zfull = torch.zeros((B, Rg_total), device=device, dtype=dtype)
            off_c = 0
            for i in idx_lam:
                s = sizes[i]
                zfull[:, ofs[i]:ofs[i]+s] = zz_compact[:, off_c:off_c+s]
                off_c += s
            return zfull

        # -------- paso trapecial dependiente de z, inyectando KKT + Baumgarte --------
        def trap_step_with_z(zc: torch.Tensor):
            from .block import ContinuousBlock, DiscreteBlock
            # (1) inbuf(t) + λ_KKT(t)
            inbuf_t, _ = self._phaseA_eval(model, states, t, externals_time_fn)
            if R_lam > 0:
                lam_kkt_t = (2.0 / (dt * dt)) * expand_to_full(zc)
                cons_t = model.compute_constraint_forces(t, states, inbuf_t, lam_kkt_t) or {}
                for bname, dports in cons_t.items():
                    dst = inbuf_t.setdefault(bname, {})
                    for pname, ten in dports.items():
                        dst[pname] = dst.get(pname, 0) + ten

            # (2) Baumgarte en t: λ_bg^t = - α·g0   (solo idx_bg)
            if len(idx_bg) > 0:
                g0_list = self._eval_constraints_list(model, t, states, inbuf_t)
                lam_list_t: List[Optional[torch.Tensor]] = [None]*len(sizes)
                for k in idx_bg:
                    g0 = g0_list[k]
                    lam_list_t[k] = - self.baumgarte_alpha * g0
                lam_bg_t_full = pack_full(lam_list_t)
                if lam_bg_t_full.numel() > 0:
                    cons_bg_t = model.compute_constraint_forces(t, states, inbuf_t, lam_bg_t_full) or {}
                    for bname, dports in cons_bg_t.items():
                        dst = inbuf_t.setdefault(bname, {})
                        for pname, ten in dports.items():
                            dst[pname] = dst.get(pname, 0) + ten

            # (3) predictor con inbuf(t) ya modificado
            pred_states: Dict[str, torch.Tensor] = {}
            for bname, blk in model.blocks.items():
                st0 = states[bname]
                if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                    dx0, _ = blk.ode(st0, inbuf_t[bname], t)
                    pred_states[bname] = st0 + dt * dx0
                elif isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                    st1p, _ = blk.update(st0, inbuf_t[bname], float(dt), float(t+dt))
                    pred_states[bname] = st1p
                else:
                    pred_states[bname] = st0

            # (4) inbuf(t+dt) + λ_KKT(t+dt)
            inbuf_t1, outs_t1 = self._phaseA_eval(model, pred_states, t + dt, externals_time_fn)
            if R_lam > 0:
                lam_kkt_t1 = (2.0 / (dt * dt)) * expand_to_full(zc)
                cons_t1 = model.compute_constraint_forces(t + dt, pred_states, inbuf_t1, lam_kkt_t1) or {}
                for bname, dports in cons_t1.items():
                    dst = inbuf_t1.setdefault(bname, {})
                    for pname, ten in dports.items():
                        dst[pname] = dst.get(pname, 0) + ten

            # (5) Baumgarte en t+dt: λ_bg^{t+dt} = - α·g1_pred - β·(g1_pred - g0)/dt  (solo idx_bg)
            if len(idx_bg) > 0:
                g0_list = self._eval_constraints_list(model, t, states, inbuf_t)             # g0 con inbuf(t)
                g1_list = self._eval_constraints_list(model, t + dt, pred_states, inbuf_t1)   # g1_pred con predictor
                lam_list_t1: List[Optional[torch.Tensor]] = [None]*len(sizes)
                for k in idx_bg:
                    g0 = g0_list[k]; g1 = g1_list[k]
                    gdot = (g1 - g0) / float(dt)
                    lam_list_t1[k] = - (self.baumgarte_alpha * g1 + self.baumgarte_beta * gdot)
                lam_bg_t1_full = pack_full(lam_list_t1)
                if lam_bg_t1_full.numel() > 0:
                    cons_bg_t1 = model.compute_constraint_forces(t + dt, pred_states, inbuf_t1, lam_bg_t1_full) or {}
                    for bname, dports in cons_bg_t1.items():
                        dst = inbuf_t1.setdefault(bname, {})
                        for pname, ten in dports.items():
                            dst[pname] = dst.get(pname, 0) + ten

            # (6) corrector trapecial con inbuf(t) y inbuf(t+dt) ya modificados
            new_states_z: Dict[str, torch.Tensor] = {k: v for k, v in states.items()}
            for bname, blk in model.blocks.items():
                st0 = states[bname]
                if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                    dx0, _ = blk.ode(st0, inbuf_t[bname], t)
                    x_star = st0 + dt * dx0
                    dx1, _ = blk.ode(x_star, inbuf_t1[bname], t + dt)
                    st1 = st0 + 0.5 * dt * (dx0 + dx1)
                    new_states_z[bname] = st1
                elif isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                    st1, _ = blk.update(st0, inbuf_t1[bname], float(dt), float(t+dt))
                    new_states_z[bname] = st1

            inbuf_end_z, outs_end_z = self._phaseA_eval(model, new_states_z, t + dt, externals_time_fn)
            return new_states_z, inbuf_t, inbuf_end_z, outs_end_z

        # -------- residuales (usados para resolver λ en KKT/KKT_mixed) --------
        def F_kkt(zc: torch.Tensor):
            st1, _, inbuf_end, _ = trap_step_with_z(zc)
            r, _ = model.build_residual(t + dt, states=st1, inbuf=inbuf_end, z=expand_to_full(zc))
            return r

        def F_baumgarte(_zc: torch.Tensor):
            # En Baumgarte puro no hay λ que resolver; residual vacío.
            return torch.zeros((B, 0), device=device, dtype=dtype)

        def F_mixed(zc: torch.Tensor):
            st1, _, inbuf_end, _ = trap_step_with_z(zc)
            # Evalúa g en t+dt y selecciona idx_lam (lo único que se resuelve con z):
            g_all = self._eval_constraints_list(model, t + dt, st1, inbuf_end)
            if len(idx_lam) == 0:
                return torch.zeros((B, 0), device=device, dtype=dtype)
            return torch.cat([g_all[i] for i in idx_lam], dim=1)

        # Elegir F a resolver
        if self.constraint_mode == "kkt":
            F_to_solve = F_kkt
        elif self.constraint_mode == "kkt_baumgarte":
            F_to_solve = F_baumgarte
        else:
            F_to_solve = F_mixed

        # Resolver z* (compacto si corresponde) y construir estado final
        if (self.constraint_mode == "kkt" and Rg_total > 0) or (self.constraint_mode == "kkt_mixed" and R_lam > 0):
            z_star = self.alg.solve(F_to_solve, z0_compact)
        else:
            z_star = z0_compact  # (B,0)
        setattr(self.alg, "last_solution", z_star)

        # Reconstruir estado/outs coherentes con z*
        new_states, _, inbuf_end, outs_end = trap_step_with_z(z_star)
        return new_states, outs_end
