# kombox/core/solvers_trapezoidal.py
from __future__ import annotations
from typing import Dict, Callable, Optional
import torch
from .solvers import SolverBase

class TrapezoidalSolver(SolverBase):
    """
    Stepper implícito tipo Trapecio (semi-implícito MVP).
    - Resuelve residual algebraico global con un AlgebraicSolver externo.
    - Integra estados continuos por-bloque con predictor-corrector trapecial usando
      entradas obtenidas por el wiring del modelo en t y t+dt.
    """
    name = "trapezoidal"
    is_global: bool = True

    def __init__(self, algebraic_solver, tol: float = 1e-8, max_iter: int = 20):
        super().__init__()
        self.alg = algebraic_solver
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def step_all(self,
                 model,
                 states: Dict[str, torch.Tensor],
                 dt: float,
                 t: float,
                 externals_time_fn: Optional[Callable[[float], Dict[str, Dict[str, torch.Tensor]]]] = None):
        from .block import ContinuousBlock, DiscreteBlock, Block

        B = next(iter(states.values())).shape[0] if states else 1
        device = next(iter(states.values())).device if states else torch.device("cpu")
        dtype = next(iter(states.values())).dtype if states else torch.float32

        downstream = model._downstream
        orderA = model._order_outputs
        ext_in = model._ext_in

        # ----- helpers Fase A -----
        def inject_externals(inbuf: Dict[str, Dict[str, torch.Tensor]], tnow: float):
            if externals_time_fn is None:
                return
            ext_map = externals_time_fn(tnow) or {}
            for d in inbuf.values():
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
                    inbuf[blk][port] = ten

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

        # 1) Resolver residual algebraico en t (MVP: con estados actuales)
        inbuf_t, _ = phaseA_eval(states, t)
        z0 = torch.zeros((B, 0), device=device, dtype=dtype)

        def F_now(z):
            # Llama a build_residual de forma compatible con firmas antiguas/nuevas
            try:
                r = model.build_residual(t, states, {k: dict(v) for k, v in inbuf_t.items()}, z=z)
            except TypeError:
                # Compatibilidad: versiones sin 'z'
                r = model.build_residual(t, states, {k: dict(v) for k, v in inbuf_t.items()})
            # Compatibilidad: si devuelve (residual, detalle), nos quedamos con el residual
            if isinstance(r, tuple):
                r = r[0]
            return r

        z_now = self.alg.solve(F_now, z0)  # noqa: F841

        # 2) Integrar continuos por-bloque (trap predictor-corrector usando ins en t y t+dt)
        new_states: Dict[str, torch.Tensor] = {k: v for k, v in states.items()}
        # Predictor continuo (Euler) para estimar ins a t+dt
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

        # Corrector trap
        for bname, blk in model.blocks.items():
            st0 = states[bname]
            if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                ins_t = inbuf_t[bname]
                ins_tp = inbuf_tp[bname]
                dx0, _ = blk.ode(st0, ins_t, t)
                x_star = st0 + dt * dx0
                dx1, _ = blk.ode(x_star, ins_tp, t+dt)
                st1 = st0 + 0.5*dt*(dx0 + dx1)
                new_states[bname] = st1
            elif isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                st1, _ = blk.update(st0, inbuf_tp[bname], float(dt), float(t+dt))
                new_states[bname] = st1

        # 3) Salidas finales coherentes con estados nuevos a t+dt
        inbuf_final, outs_final = phaseA_eval(new_states, t+dt)
        return new_states, outs_final
    