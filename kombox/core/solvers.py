# kombox/core/solvers.py
from __future__ import annotations
import torch
from typing import Tuple, Dict, Callable, Optional, Mapping

class SolverBase:
    name: str = "base"
    is_global: bool = False  # por defecto, solvers por-bloque
    def step_continuous(self, block, state: torch.Tensor, inputs, dt: float, t: float) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    # Solo para solvers "globales" (que integran TODO el sistema de una vez)
    def step_all(self, model, states: Dict[str, torch.Tensor], dt: float, t: float,
                 externals_time_fn: Optional[Callable[[float], Dict[str, Dict[str, torch.Tensor]]]] = None):
        raise NotImplementedError


class EulerSolver(SolverBase):
    name = "euler"
    def step_continuous(self, block, state, inputs, dt, t):
        dx, outs = block.ode(state, inputs, t)
        if state.numel() == 0:
            return state, outs
        new_state = state + dt * dx
        return new_state, outs


class RK4Solver(SolverBase):
    name = "rk4"
    def step_continuous(self, block, state, inputs, dt, t):
        if state.numel() == 0:
            _, outs = block.ode(state, inputs, t)
            return state, outs
        k1, _ = block.ode(state, inputs, t)
        k2, _ = block.ode(state + 0.5*dt*k1, inputs, t + 0.5*dt)
        k3, _ = block.ode(state + 0.5*dt*k2, inputs, t + 0.5*dt)
        k4, outs = block.ode(state + dt*k3, inputs, t + dt)
        new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return new_state, outs


class RK45Solver(SolverBase):
    name = "rk45"

    def __init__(self, error_norm: str = "linf"):
        self.error_norm = error_norm
        self.last_error: torch.Tensor | None = None

    def step_continuous(self, block, state, inputs, dt, t):
        if state.numel() == 0:
            _, outs = block.ode(state, inputs, t + dt)
            self.last_error = None
            return state, outs
        # Dormand–Prince 5(4)
        k1, _ = block.ode(state, inputs, t)
        s2 = state + dt*(1/5)*k1;              k2, _ = block.ode(s2, inputs, t + (1/5)*dt)
        s3 = state + dt*((3/40)*k1 + (9/40)*k2); k3, _ = block.ode(s3, inputs, t + (3/10)*dt)
        s4 = state + dt*((44/45)*k1 + (-56/15)*k2 + (32/9)*k3); k4, _ = block.ode(s4, inputs, t + (4/5)*dt)
        s5 = state + dt*((19372/6561)*k1 + (-25360/2187)*k2 + (64448/6561)*k3 + (-212/729)*k4); k5, _ = block.ode(s5, inputs, t + (8/9)*dt)
        s6 = state + dt*((9017/3168)*k1 + (-355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 + (-5103/18656)*k5); k6, _ = block.ode(s6, inputs, t + dt)
        s7 = state + dt*((35/384)*k1 + 0*k2 + (500/1113)*k3 + (125/192)*k4 + (-2187/6784)*k5 + (11/84)*k6)
        k7, outs_end = block.ode(s7, inputs, t + dt)

        y5 = s7
        y4 = state + dt*((5179/57600)*k1 + 0*k2 + (7571/16695)*k3 + (393/640)*k4
                         + (-92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7)
        err = (y5 - y4)
        if self.error_norm == "linf":
            self.last_error = err.abs().amax(dim=1, keepdim=True)
        elif self.error_norm == "l2":
            self.last_error = err.pow(2).sum(dim=1, keepdim=True).sqrt()
        else:
            self.last_error = err
        return y5, outs_end


# -------------------------- TorchDiffEq (global) ------------------------------
class TorchDiffEqSolver(SolverBase):
    """
    Adaptador a torchdiffeq (odeint / odeint_adjoint) que integra el
    SISTEMA CONTINUO (ContinuousBlock) en un único vector, reevaluando:
      - wiring Fase A (outs) en cada subpaso interno,
      - externals_fn(t) a tiempo continuo.

    MODO HÍBRIDO:
      - Admite bloques DiscreteBlock con estado: se consideran piecewise-constant
        durante (t, t+dt). Tras integrar, se actualizan con update(dt, t+dt).
      - Tras actualizar los discretos, se recalculan salidas (Fase A) coherentes a t+dt.

    Nota: Los bloques algebraicos (S=0) se evalúan en Fase A, no se integran.
    """
    name = "torchdiffeq"
    is_global: bool = True

    def __init__(self, method: str = "dopri5", use_adjoint: bool = False,
                 rtol: float = 1e-5, atol: float = 1e-7):
        try:
            if use_adjoint:
                from torchdiffeq import odeint_adjoint as odeint
            else:
                from torchdiffeq import odeint
        except Exception as e:
            raise ImportError(
                "TorchDiffEqSolver requiere 'torchdiffeq'. Instala con:\n"
                "  pip install torchdiffeq"
            ) from e
        self._odeint = odeint
        self.method = method
        self.rtol = float(rtol)
        self.atol = float(atol)

    # ---------- empaquetado continuo (solo ContinuousBlock con S>0) ------------
    def _make_packers(self, model, states: Dict[str, torch.Tensor]):
        from .block import ContinuousBlock
        cont_names: list[str] = []
        sizes: list[int] = []
        for name, blk in model.blocks.items():
            if isinstance(blk, ContinuousBlock) and blk.state_size() > 0:
                cont_names.append(name)
                sizes.append(int(blk.state_size()))

        slices: dict[str, slice] = {}
        off = 0
        for name, S in zip(cont_names, sizes):
            sl = slice(off, off + S); slices[name] = sl; off += S
        total_S = off

        B = next(iter(states.values())).shape[0]
        device = next(iter(states.values())).device
        dtype = next(iter(states.values())).dtype

        def pack(sts: Dict[str, torch.Tensor]) -> torch.Tensor:
            if not cont_names:
                return torch.empty((B, 0), device=device, dtype=dtype)
            cols = [sts[name] for name in cont_names]
            return torch.cat(cols, dim=1)

        def unpack(xflat: torch.Tensor, base_states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            # devuelve un dict con: continuos de xflat, discretos/algebraicos copiando de base_states
            out: Dict[str, torch.Tensor] = {}
            # primero copia todo desde base
            for n, st in base_states.items():
                out[n] = st
            # luego sustituye continuos
            for name, sl in slices.items():
                out[name] = xflat[:, sl]
            return out

        return cont_names, sizes, slices, total_S, pack, unpack
    # ---------- inyectar externals_fn(t) en inbuf ------------------------------
    def _inject_externals(self, model, inbuf: dict[str, dict[str, torch.Tensor]], eff_ext: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        """
        Inyecta entradas externas de forma estricta:
        - Para cada conexión external 'ext_name' -> (blk_name, port_can), exige que eff_ext[ext_name]
          contenga una clave exactamente igual al puerto canónico 'port_can' o a alguno de sus alias.
        - Si no aparece ni canónico ni alias, lanza KeyError con un mensaje claro.
        """
        # 1) limpiar buffer de entradas
        for d in inbuf.values():
            d.clear()

        # 2) inyectar una a una
        for ext_name, targets in model._ext_in.items():
            if ext_name not in eff_ext:
                raise KeyError(
                    f"External '{ext_name}' no proporcionado por externals/external_fn. "
                    f"Esperados: {sorted(model._ext_in.keys())}"
                )
            src_dict = eff_ext[ext_name]
            if not isinstance(src_dict, Mapping):
                raise TypeError(
                    f"External '{ext_name}' debe ser un dict puerto->tensor, recibido {type(src_dict).__name__}"
                )

            for (blk_name, port_can) in targets:
                blk = model.blocks[blk_name]

                # Construir conjunto de nombres aceptados: canónico + alias que mapeen a ese canónico
                accepted = {port_can}
                # Compatibilidad con distintos nombres internos de mapa de alias
                alias_maps = []
                if hasattr(blk, "_in_alias"):      # alias -> canónico
                    alias_maps.append(getattr(blk, "_in_alias"))
                if hasattr(blk, "_input_alias"):   # alias -> canónico (otra variante)
                    alias_maps.append(getattr(blk, "_input_alias"))
                if hasattr(blk, "_aliases_in"):    # alias -> canónico (otra variante)
                    alias_maps.append(getattr(blk, "_aliases_in"))

                for amap in alias_maps:
                    if isinstance(amap, dict):
                        for alias, canon in amap.items():
                            if canon == port_can:
                                accepted.add(alias)

                # Buscar clave válida en eff_ext[ext_name]
                key_found = next((k for k in accepted if k in src_dict), None)
                if key_found is None:
                    raise KeyError(
                        f"External '{ext_name}': falta la clave '{port_can}'. "
                        f"Claves aceptadas para ese destino: {sorted(accepted)}. "
                        f"Claves disponibles: {sorted(src_dict.keys())}"
                    )

                ten = src_dict[key_found]
                inbuf[blk_name][port_can] = ten

    # ---------- paso global ----------------------------------------------------
    def step_all(self, model, states: Dict[str, torch.Tensor], dt: float, t: float,
                 externals_time_fn: Optional[Callable[[float], Dict[str, Dict[str, torch.Tensor]]]] = None):

        from .block import Block, ContinuousBlock, DiscreteBlock

        names, sizes, slices, total_S, pack, unpack = self._make_packers(model, states)
        B = next(iter(states.values())).shape[0]
        device = next(iter(states.values())).device
        dtype = next(iter(states.values())).dtype

        downstream = model._downstream
        orderA = model._order_outputs
        ext_in = model._ext_in

        # ----- helpers Fase A a tiempo continuo -----
        def phaseA_eval(states_dict: Dict[str, torch.Tensor], tnow: float):
            inbuf: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in model.blocks.keys()}
            ext_map = externals_time_fn(tnow) if externals_time_fn else {}
            self._inject_externals(model, inbuf, ext_map)
            outs_cache: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in model.blocks.keys()}
            for bname in model._order_outputs:
                blk = model.blocks[bname]
                st  = states_dict[bname]
                ins = inbuf[bname]
                outs = blk._expose_outputs(st, ins, tnow)
                outs_cache[bname] = outs
                # propaga por wiring
                for (port_name, ten) in outs.items():
                    for (dst_blk, dst_port) in model._downstream.get((bname, port_name), []):
                        inbuf[dst_blk][dst_port] = ten
            return inbuf, outs_cache

        # ----- campo vectorial g(t, xflat) para torchdiffeq -----
        def g(t_tensor: torch.Tensor, xflat: torch.Tensor) -> torch.Tensor:
            tnow = float(t_tensor.item())
            cand_states = unpack(xflat, states)  # cont actualizados, discretos/algebraicos constantes
            inbuf, _ = phaseA_eval(cand_states, tnow)

            # Derivadas solo de bloques continuos con S>0
            dx_cols = []
            for bname in names:
                blk: ContinuousBlock = model.blocks[bname]
                st = cand_states[bname]
                ins = inbuf[bname]
                dx, _ = blk.ode(st, ins, tnow)
                dx_cols.append(dx)
            if not dx_cols:
                return torch.empty((B, 0), device=device, dtype=dtype)
            return torch.cat(dx_cols, dim=1)

        # ----- integra (t, t+dt) -----
        x0 = pack(states)  # (B, total_S) o (B,0)
        tt = torch.tensor([t, t + dt], device=device, dtype=dtype)
        y = self._odeint(g, x0, tt, method=self.method, rtol=self.rtol, atol=self.atol)  # (2, B, total_S)
        x1 = y[-1]
        new_states = unpack(x1, states)  # continuos a t+dt, discretos todavía “viejos”

        # ----- actualizar discretos con entradas a t+dt -----
        inbuf_end, outs_end = phaseA_eval(new_states, t + dt)  # usa estados continuos nuevos
        overrides: Dict[str, Dict[str, torch.Tensor]] = {}

        for bname, blk in model.blocks.items():
            if isinstance(blk, DiscreteBlock) and blk.state_size() > 0:
                st_old = new_states[bname]  # aún es el estado “viejo”
                ins = inbuf_end[bname]
                st_new, outsD = blk.update(st_old, ins, float(dt), float(t + dt))
                new_states[bname] = st_new
                overrides[bname] = outsD

        # ----- Fase A final coherente con TODOS los estados nuevos -----
        inbuf_final, outs_final = phaseA_eval(new_states, t + dt)
        
        for bname in orderA:
            blk: Block = model.blocks[bname]
            st = new_states[bname]
            ins = inbuf_final[bname]
            if bname in overrides:
                outs = overrides[bname]
            else:
                outs = blk._expose_outputs(st, ins, t + dt)
            outs_final[bname] = outs
            for (port_name, ten) in outs.items():
                for (dst_blk, dst_port) in downstream.get((bname, port_name), []):
                    inbuf_final[dst_blk][dst_port] = ten

        return new_states, outs_final
