# kombox/core/solvers_trapezoidal.py
from __future__ import annotations
from typing import Dict, Callable, Optional, Tuple, List
import torch
from .solvers import SolverBase
from .simulator import _flatten_states, _unflatten_states, _collect_global_constraints
from .algebraic.newton_krylov import NewtonKrylov  # import correcto

_ALLOWED_MODES = {"kkt", "kkt_baumgarte"}

class TrapezoidalSolver(SolverBase):
    """
    Stepper implícito tipo Trapecio (global) con acoplo KKT por λ:
    - Construye wiring en t y t+dt.
    - Suma fuerzas de restricción Φ_q^T λ (via Model.compute_constraint_forces) en ambos tiempos.
    - Integra continuos con predictor-corrector trapecial dependiente de z (λ).
    - Resuelve el residual global en t+dt con un AlgebraicSolver (p.ej., NewtonKrylov).
    - (Opcional) Estabilización de restricciones con Baumgarte.
    """
    name = "trapezoidal"
    is_global: bool = True

    def __init__(
        self,
        algebraic_solver: Optional[NewtonKrylov] = None,
        *,
        constraint_mode: Optional[str] = None,
        # compatibilidad hacia atrás (si alguien sigue pasando baumgarte_enabled)
        baumgarte_enabled: Optional[bool] = None,
        baumgarte_alpha: float = 0.0,
        baumgarte_beta: float = 0.0,
        tol: float = 1e-8,
        max_iter: int = 20,
    ):
        """
        constraint_mode:
          - "kkt"            -> restricciones en t+dt vía KKT puro
          - "kkt_baumgarte"  -> KKT con g* = g + α·dt·g_dot + β·dt²·g (Baumgarte)
        Si 'constraint_mode' no se pasa pero se pasa 'baumgarte_enabled=True',
        se asume "kkt_baumgarte" para compatibilidad.
        """
        super().__init__()
        self.alg = algebraic_solver or NewtonKrylov()
        self.is_global = True

        # Resolver modo con compatibilidad hacia atrás
        if constraint_mode is None:
            if baumgarte_enabled is True:
                constraint_mode = "kkt_baumgarte"
            else:
                constraint_mode = "kkt"
        if constraint_mode not in _ALLOWED_MODES:
            raise ValueError(f"constraint_mode debe ser uno de {_ALLOWED_MODES}, got {constraint_mode!r}")
        self.constraint_mode = constraint_mode

        self.baumgarte_alpha = float(baumgarte_alpha)
        self.baumgarte_beta  = float(baumgarte_beta)
        # flag interno: si el modo es kkt_baumgarte, activamos Baumgarte
        self._baumgarte_on = (self.constraint_mode == "kkt_baumgarte")
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def step_all(
        self,
        model,
        states: Dict[str, torch.Tensor],
        dt: float,
        t: float,
        externals_time_fn: Optional[Callable[[float], Dict[str, Dict[str, torch.Tensor]]]] = None
    ):
        from .block import ContinuousBlock, DiscreteBlock, Block

        B = next(iter(states.values())).shape[0] if states else 1
        device = next(iter(states.values())).device if states else torch.device("cpu")
        dtype  = next(iter(states.values())).dtype  if states else torch.float32

        downstream = model._downstream
        orderA = model._order_outputs
        ext_in = model._ext_in

        # -------- helpers: wiring (Fase A) --------
        def inject_externals(into_inbuf: Dict[str, Dict[str, torch.Tensor]], tnow: float):
            if externals_time_fn is None:
                # limpiar igualmente
                for d in into_inbuf.values():
                    d.clear()
                return
            ext_map = externals_time_fn(tnow) or {}
            for d in into_inbuf.values():
                d.clear()
            for ext_name, targets in ext_in.items():
                if ext_name not in ext_map:
                    raise KeyError(f"Falta entrada externa '{ext_name}' en externals_fn(t={tnow:.6f}).")
                src_dict = ext_map[ext_name]
                for (blk, port) in targets:
                    if port in src_dict:
                        ten = src_dict[port]
                    elif len(src_dict) == 1:
                        ten = next(iter(src_dict.values()))
                    elif ext_name in src_dict and isinstance(src_dict[ext_name], torch.Tensor):
                        ten = src_dict[ext_name]
                    else:
                        raise KeyError(f"External '{ext_name}' no provee puerto '{port}'.")
                    into_inbuf[blk][port] = ten

        def phaseA_eval(states_dict: Dict[str, torch.Tensor], tnow: float):
            inbuf: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in model.blocks.keys()}
            inject_externals(inbuf, tnow)
            outs_cache: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in model.blocks.keys()}
            for bname in orderA:
                blk: Block = model.blocks[bname]
                st = states_dict[bname]
                ins = inbuf[bname]
                outs = blk._expose_outputs(st, ins, tnow)
                outs_cache[bname] = outs
                for (port_name, ten) in outs.items():
                    for (dst_blk, dst_port) in downstream.get((bname, port_name), []):
                        inbuf[dst_blk][dst_port] = ten
            return inbuf, outs_cache

        # -------- tamaño de z (Rg) desde g(t,·) --------
        inbuf_t0, _ = phaseA_eval(states, t)
        g0 = _collect_global_constraints(model, t, states, inbuf_t0, z=None)  # (B,Rg)
        Rg = int(g0.shape[1]) if g0.ndim == 2 else 0
        z0 = torch.zeros((B, Rg), device=device, dtype=dtype)  # si Rg=0 => (B,0)

        # -------- función de paso trapecial dependiente de z --------
        def trap_step_with_z(zz: torch.Tensor):
            """
            Construye inbuf en t y t+dt, suma Φ_q^T λ, evalúa ODEs y devuelve:
            (states_end_z, inbuf_end_z, outs_end_z).
            """
            # inbuf(t) + fuerzas de restricción(t)
            inbuf_t, _ = phaseA_eval(states, t)
            if zz is not None and zz.numel() > 0:
                # Reparam: λ = (2/dt^2) * z  ⇒  x_{k+1} ≈ x_k + z   (mejor condicionamiento)
                lam_eff = (2.0 / (dt * dt)) * zz
                cons_t = model.compute_constraint_forces(t, states, inbuf_t, lam_eff) or {}
                for bname, dports in cons_t.items():
                    dst = inbuf_t.setdefault(bname, {})
                    for pname, ten in dports.items():
                        dst[pname] = dst.get(pname, 0) + ten

            # predictor con dx0(z)
            pred_states: Dict[str, torch.Tensor] = {}
            for bname, blk in model.blocks.items():
                st0 = states[bname]
                if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                    ins_t = inbuf_t[bname]
                    dx0, _ = blk.ode(st0, ins_t, t)
                    pred_states[bname] = st0 + dt * dx0
                elif isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                    # predictor discreto: usar update hacia t+dt con inbuf(t) (aprox)
                    st1p, _ = blk.update(st0, inbuf_t[bname], float(dt), float(t+dt))
                    pred_states[bname] = st1p
                else:
                    pred_states[bname] = st0

            # inbuf(t+dt) (con pred_states) + fuerzas de restricción(t+dt)
            inbuf_t1, outs_t1 = phaseA_eval(pred_states, t + dt)
            if zz is not None and zz.numel() > 0:
                lam_eff1 = (2.0 / (dt * dt)) * zz
                cons_t1 = model.compute_constraint_forces(t + dt, pred_states, inbuf_t1, lam_eff1) or {}
                for bname, dports in cons_t1.items():
                    dst = inbuf_t1.setdefault(bname, {})
                    for pname, ten in dports.items():
                        dst[pname] = dst.get(pname, 0) + ten

            # corrector trapecial con inbuf(z)
            new_states_z: Dict[str, torch.Tensor] = {k: v for k, v in states.items()}
            for bname, blk in model.blocks.items():
                st0 = states[bname]
                if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                    ins_t  = inbuf_t[bname]
                    ins_t1 = inbuf_t1[bname]
                    dx0, _ = blk.ode(st0, ins_t, t)
                    x_star = st0 + dt * dx0
                    dx1, _ = blk.ode(x_star, ins_t1, t + dt)
                    st1 = st0 + 0.5 * dt * (dx0 + dx1)
                    new_states_z[bname] = st1
                elif isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                    st1, _ = blk.update(st0, inbuf_t1[bname], float(dt), float(t+dt))
                    new_states_z[bname] = st1

            inbuf_end_z, outs_end_z = phaseA_eval(new_states_z, t + dt)
            return new_states_z, inbuf_end_z, outs_end_z

        # -------- residual en t+dt dependiente de z (con Baumgarte opcional) --------
        def F_end(zz: torch.Tensor):
            st1, inbuf_end, _ = trap_step_with_z(zz)
            r = model.build_residual(t + dt, states=st1, inbuf=inbuf_end, z=zz)
            return r[0] if isinstance(r, tuple) else r

        # Elegir residual final según modo
        if self.constraint_mode == "kkt":
            F_to_solve = F_end
        else:  # "kkt_baumgarte"
            def F_end_baumgarte(z):
                st1, inbuf_end, _ = trap_step_with_z(z)
                # aplicar Baumgarte sobre el residual de restricciones globales
                r_aug, _ = self._apply_baumgarte(
                    model,
                    t_end=t + dt,
                    dt=dt,
                    states_end=st1,
                    inbuf_end=inbuf_end,
                    z_end=z,
                )
                return r_aug
            F_to_solve = F_end_baumgarte

        # Resolver z(t+dt) y USAR la solución devuelta por el solver
        if Rg > 0:
            z_star = self.alg.solve(F_to_solve, z0)
        else:
            z_star = z0

        # (opcional) guarda por compatibilidad si tu NewtonKrylov no lo hace
        setattr(self.alg, "last_solution", z_star)

        # Reconstruir estado y salidas coherentes con z*
        new_states, inbuf_end, outs_end = trap_step_with_z(z_star)

        return new_states, outs_end

    # ---------- Baumgarte ----------
    def _apply_baumgarte(
        self,
        model,
        *,
        t_end: float,
        dt: float,
        states_end: Dict[str, torch.Tensor],
        inbuf_end: Dict[str, Dict[str, torch.Tensor]],
        z_end: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Crea un residual 'reconstruido' donde las restricciones globales han sido
        estabilizadas con Baumgarte:
            g* = g + α dt g_dot + β dt^2 g
        con  g_dot ≈ J_g(x) · f(x)  (vía autograd).
        """
        r_raw, detail = model.build_residual(t_end, states=states_end, inbuf=inbuf_end, z=z_end)
        B = r_raw.shape[0] if r_raw.ndim == 2 else 1
        dtype = r_raw.dtype
        device = r_raw.device

        # Extraer g(x)
        g = _collect_global_constraints(model, t_end, states_end, inbuf_end, z_end)
        # si no hay restricciones globales, o el modo no es Baumgarte, o α/β = 0 → devolvemos residual original
        if (
            g.shape[1] == 0
            or not self._baumgarte_on
            or (self.baumgarte_alpha == 0.0 and self.baumgarte_beta == 0.0)
        ):
            # reconstruye residual original (sin modificación)
            pieces: List[torch.Tensor] = []
            for bname in model.blocks.keys():
                if bname in detail:
                    vv = detail[bname]; vv = vv if vv.ndim == 2 else vv.view(B, -1)
                    pieces.append(vv)
            for (cname, _) in model._constraints:
                key = f"__global__::{cname}"
                if key in detail:
                    vv = detail[key]; vv = vv if vv.ndim == 2 else vv.view(B, -1)
                    pieces.append(vv)
            return (torch.cat(pieces, dim=1) if len(pieces) > 0 else torch.zeros((B, 0), dtype=dtype, device=device)), detail

        # Aplanar estados y construir f(x) en t_end
        Xall, metas = _flatten_states(states_end)
        if Xall.shape[1] == 0:
            pieces = []
            for bname in model.blocks.keys():
                if bname in detail: pieces.append(detail[bname])
            for (cname, _) in model._constraints:
                key = f"__global__::{cname}"
                if key in detail: pieces.append(detail[key])
            return (torch.cat(pieces, dim=1) if len(pieces) > 0 else torch.zeros((B, 0), dtype=dtype, device=device)), detail

        # f_end por concatenación de dx/dt
        f_cols = []
        for bname, blk in model.blocks.items():
            st = states_end[bname]
            ins = inbuf_end.get(bname, {})
            dx, _ = blk.ode(st, ins, t_end)
            f_cols.append(dx if dx.ndim == 2 else dx.view(B, -1))
        f_end = torch.cat([c for c in f_cols if c.numel() > 0], dim=1) if len(f_cols) > 0 else torch.zeros_like(Xall)

        # g(x) y J_g(x) batched (sin in-place)
        def g_only(Xflat: torch.Tensor) -> torch.Tensor:
            st = _unflatten_states(Xflat, metas, states_end)
            return _collect_global_constraints(model, t_end, st, inbuf_end, z_end)  # (B,Rg)

        Xvar = Xall.clone().requires_grad_(True)
        g_now = g_only(Xvar)  # (B,Rg)
        J_full = torch.autograd.functional.jacobian(g_only, Xvar, create_graph=True)  # (B,Rg,B,S)

        alpha = self.baumgarte_alpha
        beta  = self.baumgarte_beta
        S = Xvar.shape[1]
        Rg = g_now.shape[1]

        # g_dot = J f  (por batch)
        gdot_list = []
        for b in range(B):
            Jb = J_full[b, :, b, :]            # (Rg,S)
            fb = f_end[b].view(S, 1)           # (S,1)
            gdot_b = (Jb @ fb).view(1, Rg)     # (1,Rg)
            gdot_list.append(gdot_b)
        gdot = torch.cat(gdot_list, dim=0)     # (B,Rg)

        # g* = g + α dt g_dot + β dt^2 g
        g_star = g_now + alpha * dt * gdot + beta * (dt ** 2) * g_now

        # Reconstruir residual: bloques + g*
        pieces = []
        for bname in model.blocks.keys():
            if bname in detail:
                vv = detail[bname]; vv = vv if vv.ndim == 2 else vv.view(B, -1)
                pieces.append(vv)
        if Rg > 0:
            pieces.append(g_star)
        r_aug = torch.cat(pieces, dim=1) if len(pieces) > 0 else torch.zeros((B, 0), dtype=dtype, device=device)
        return r_aug, detail
    
    def _baumgarte_residual(self,
                            model,
                            *,
                            t_begin: float,
                            states_begin: dict,
                            inbuf_begin: dict,
                            t_end: float,
                            states_end: dict,
                            inbuf_end: dict,
                            z_end: torch.Tensor,
                            dt: float) -> torch.Tensor:
        """
        Residual de Baumgarte a tiempo n+1:
            F_BG = gdot^{n+1} + alpha * g^{n+1}
        con gdot^{n+1} ≈ (g^{n+1} - g^{n}) / dt

        Devuelve un tensor (B, Rg).
        """
        # g^n (no depende de z_end)
        g_begin, _ = model.build_residual(t_begin, states_begin, inbuf_begin, z=None)
        # g^{n+1} (sí puede depender de z_end vía estados_end/inbuf_end)
        g_end,   _ = model.build_residual(t_end,   states_end,   inbuf_end,   z=z_end)

        gdot = (g_end - g_begin) / float(dt)
        Fbg  = gdot + self.baumgarte_alpha * g_end
        return Fbg

