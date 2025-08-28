# kombox/core/solvers_trapezoidal.py
from __future__ import annotations
from typing import Dict, Callable, Optional, Tuple, List
import torch
from .solvers import SolverBase
from .simulator import _flatten_states, _unflatten_states, _collect_global_constraints
from .algebraic.newton_krylov import NewtonKrylov  # import correcto

class TrapezoidalSolver(SolverBase):
    """
    Stepper implícito tipo Trapecio (global):
    - Fase A: wiring en t y t+dt para estimar entradas.
    - Integra continuos por-bloque con predictor-corrector trapecial.
    - Resuelve el residual algebraico global en t+dt con un AlgebraicSolver (p.ej., NewtonKrylov).
    - (Opcional) Estabilización de restricciones globales con Baumgarte.
    """
    name = "trapezoidal"
    is_global: bool = True

    def __init__(
        self,
        algebraic_solver: Optional[NewtonKrylov] = None,
        *,
        tol: float = 1e-8,
        max_iter: int = 20,
        baumgarte_alpha: float = 0.0,
        baumgarte_beta: float = 0.0,
        baumgarte_enabled: bool = False,
    ):
        super().__init__()
        self.alg = algebraic_solver or NewtonKrylov()
        self.is_global = True
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        # Baumgarte
        self.baumgarte_enabled = bool(baumgarte_enabled)
        self.baumgarte_alpha = float(baumgarte_alpha)
        self.baumgarte_beta  = float(baumgarte_beta)

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

        downstream = model._downstream
        orderA = model._order_outputs
        ext_in = model._ext_in

        # ----- helpers Fase A (wiring) -----
        def inject_externals(into_inbuf: Dict[str, Dict[str, torch.Tensor]], tnow: float):
            if externals_time_fn is None:
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

        # ---------- 1) Residual en t (opcional/diagnóstico) ----------
        inbuf_t, _ = phaseA_eval(states, t)
        z0 = torch.zeros((B, 0), device=next(iter(states.values())).device, dtype=next(iter(states.values())).dtype)

        def F_now(z):
            r = model.build_residual(t, states, {k: dict(v) for k, v in inbuf_t.items()}, z=z)
            return r[0] if isinstance(r, tuple) else r

        # Resolver en t (útil si hay z(t) o diagnósticos; no forzamos uso posterior)
        _ = self.alg.solve(F_now, z0)

        # ---------- 2) Predictor para estimar inbuf en t+dt ----------
        pred_states: Dict[str, torch.Tensor] = {}
        for bname, blk in model.blocks.items():
            st0 = states[bname]
            if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                ins_t = inbuf_t[bname]
                dx0, _ = blk.ode(st0, ins_t, t)
                pred_states[bname] = st0 + dt * dx0
            else:
                pred_states[bname] = st0

        inbuf_tp, _ = phaseA_eval(pred_states, t+dt)

        # ---------- 3) Corrector trapecial ----------
        new_states: Dict[str, torch.Tensor] = {k: v for k, v in states.items()}
        for bname, blk in model.blocks.items():
            st0 = states[bname]
            if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                ins_t = inbuf_t[bname]
                ins_tp = inbuf_tp[bname]
                dx0, _ = blk.ode(st0, ins_t, t)
                x_star = st0 + dt * dx0
                dx1, _ = blk.ode(x_star, ins_tp, t+dt)
                st1 = st0 + 0.5 * dt * (dx0 + dx1)
                new_states[bname] = st1
            elif isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                st1, _ = blk.update(st0, inbuf_tp[bname], float(dt), float(t+dt))
                new_states[bname] = st1

        # ---------- 4) Residual en t+dt (con Baumgarte opcional) ----------
        t_end = t + dt
        inbuf_end, outs_end = phaseA_eval(new_states, t_end)

        def F_end(zz):
            r = model.build_residual(t_end, states=new_states, inbuf=inbuf_end, z=zz)
            return r[0] if isinstance(r, tuple) else r

        if self.baumgarte_enabled:
            # Envolver F_end para sustituir las restricciones globales por g* (Baumgarte)
            def F_end_baumgarte(zz):
                r_aug, _ = self._apply_baumgarte(
                    model,
                    t_end=t_end,
                    dt=dt,
                    states_end=new_states,
                    inbuf_end=inbuf_end,
                    z_end=zz,
                )
                return r_aug
            F_to_solve = F_end_baumgarte
        else:
            F_to_solve = F_end

        _ = self.alg.solve(F_to_solve, z0)  # resuelve z(t+dt); útil para DAE/LM

        # ---------- 5) Devolver estados y salidas a t+dt ----------
        # (outs_end coherentes con new_states en t+dt)
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
        if g.shape[1] == 0 or not self.baumgarte_enabled or (self.baumgarte_alpha == 0.0 and self.baumgarte_beta == 0.0):
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
