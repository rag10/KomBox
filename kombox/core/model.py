# kombox/core/model.py
from __future__ import annotations
from typing import Dict, Mapping, Tuple, Optional, List
from collections import deque
import torch
import torch.nn as nn

from .block import Block

_EXT = "__external__"


class Model(Block):
    """
    Modelo compuesto (anidable). Ahora mantiene `self._states` internamente.
    Flujo típico:
        m.add_block(...); m.connect(...); m.build()
        m.initialize(B, device=..., dtype=..., initial_states=...)
        sim = Simulator(m)  # sin pasar states
    """

    def __init__(self, name: str = "model"):
        super().__init__()
        self.name = name
        self.blocks = nn.ModuleDict()

        # Conexiones en bruto (pueden usar alias); se normalizan en build()
        self._raw_connections: List[Tuple[str, str, str, str]] = []
        self.connections: List[Tuple[str, str, str, str]] = []

        # Artefactos build()
        self._built: bool = False
        self._feedthrough_cache: Dict[str, bool] = {}
        self._order_outputs: List[str] = []
        self._downstream: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self._ext_in: Dict[str, List[Tuple[str, str]]] = {}
        self._ext_out: Dict[str, Tuple[str, str]] = {}

        # Estados gestionados por el propio modelo
        self._states: Optional[Dict[str, torch.Tensor]] = None
        self._B: Optional[int] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # --- DAE/Restricciones (opt-in) ---
        self._eq_constraints: Dict[str, callable] = {}
        self._compl_constraints: Dict[str, tuple] = {}  # name -> (c_fn, lambda_name)
        self._z_layout: Optional[Dict[str, object]] = None  # planificación de variables algebraicas

# ---------------- estructura ----------------
    def add_block(self, name: str, block: Block):
        if name in self.blocks:
            raise KeyError(f"Bloque '{name}' ya existe.")
        self.blocks[name] = block
        return block

    def connect(self, src: str, dst: str):
        def parse(ep: str):
            if "." in ep:
                blk, port = ep.split(".", 1)
                if blk not in self.blocks:
                    raise KeyError(f"Bloque '{blk}' no existe.")
                return blk, port
            return _EXT, ep
        sblk, sport = parse(src); dblk, dport = parse(dst)
        self._raw_connections.append((sblk, sport, dblk, dport))
        return self

    def connect_from_strings(self, *arrows: str):
        for arrow in arrows:
            left, right = arrow.split("->")
            self.connect(left.strip(), right.strip())
        return self

    # ---------------- build (normaliza alias, ordena fases, externos) ----------------
    @torch.no_grad()
    def _infer_block_direct_feedthrough(self, name: str, blk: Block, *, tol: float = 1e-8) -> bool:
        if name in self._feedthrough_cache:
            return self._feedthrough_cache[name]
        try:
            p0 = next(blk.parameters()); device = p0.device
        except StopIteration:
            device = torch.device("cpu")
        dtype = torch.float32
        B = 1
        st = blk.init_state(B, device=device, dtype=dtype)
        if not blk.in_specs:
            self._feedthrough_cache[name] = False
            return False
        zins, rins = {}, {}
        for in_name, spec in blk.in_specs.items():
            zins[in_name] = torch.zeros((B, spec.n), device=device, dtype=dtype)
            rins[in_name] = torch.randn((B, spec.n), device=device, dtype=dtype) * 0.37
        outs0 = blk._expose_outputs(st, zins, 0.0)
        outs1 = blk._expose_outputs(st, rins, 0.0)
        changed = False
        for oname, spec in blk.out_specs.items():
            y0, y1 = outs0[oname], outs1[oname]
            if y0.shape != (B, spec.n) or y1.shape != (B, spec.n): changed = True; break
            if (y1 - y0).abs().max().item() > tol: changed = True; break
        self._feedthrough_cache[name] = changed
        return changed

    def _toposort_or_cycle(self, nodes: List[str], edges: List[Tuple[str, str]]) -> List[str]:
        adj = {u: [] for u in nodes}; indeg = {u: 0 for u in nodes}
        for u, v in edges: adj[u].append(v); indeg[v] += 1
        from collections import deque
        q = deque([u for u in nodes if indeg[u] == 0]); out: List[str] = []
        while q:
            u = q.popleft(); out.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        if len(out) == len(nodes): return out
        # reconstrucción de ciclo simple
        color = {u: 0 for u in nodes}; stack: List[str] = []
        def dfs(u: str):
            color[u] = 1; stack.append(u)
            for v in adj[u]:
                if color[v] == 0:
                    if dfs(v): return True
                elif color[v] == 1:
                    cyc = [v]; i = len(stack)-1
                    while i >= 0 and stack[i] != v: cyc.append(stack[i]); i -= 1
                    cyc.append(v); cyc.reverse()
                    raise ValueError("Algebraic loop detected (purely algebraic cycle): " +
                                     " -> ".join(cyc) +
                                     ". Inserta integrador/masa/retardo o usa solver implícito.")
            stack.pop(); color[u] = 2; return False
        for u in nodes:
            if color[u] == 0: dfs(u)
        raise ValueError("Algebraic loop detected in feedthrough subgraph.")

    def _normalize_connections(self):
        self.connections = []
        def avail(blk: Block, direction: str) -> str:
            if direction == "out":
                cans = sorted(blk.out_specs.keys()); als = sorted(blk._out_alias.keys())
            else:
                cans = sorted(blk.in_specs.keys());  als = sorted(blk._in_alias.keys())
            return f"canónicos={cans}, alias={als}"
        for sblk, sport, dblk, dport in self._raw_connections:
            # src
            if sblk == _EXT:
                s_can = sport
            else:
                blk = self.blocks[sblk]
                s_can = blk._canon_out(sport)
                if s_can not in blk.out_specs:
                    raise KeyError(f"{sblk}: salida '{sport}' no existe (→'{s_can}'). {avail(blk,'out')}")
            # dst
            if dblk == _EXT:
                d_can = dport
            else:
                blk = self.blocks[dblk]
                d_can = blk._canon_in(dport)
                if d_can not in blk.in_specs:
                    raise KeyError(f"{dblk}: entrada '{dport}' no existe (→'{d_can}'). {avail(blk,'in')}")
            self.connections.append((sblk, s_can, dblk, d_can))

    def build(self):
        self._normalize_connections()
        nodes_insertion = list(self.blocks.keys())
        ft_nodes = {n for n, b in self.blocks.items() if self._infer_block_direct_feedthrough(n, b)}
        ft_edges: List[Tuple[str, str]] = []
        for sblk, sport, dblk, dport in self.connections:
            if sblk != _EXT and dblk != _EXT and sblk in ft_nodes and dblk in ft_nodes:
                ft_edges.append((sblk, dblk))
        ft_nodes_list = sorted(ft_nodes, key=lambda n: nodes_insertion.index(n)) if ft_nodes else []
        ft_order = self._toposort_or_cycle(ft_nodes_list, ft_edges) if ft_nodes_list else []
        non_ft = [n for n in nodes_insertion if n not in ft_nodes]
        self._order_outputs = non_ft + ft_order

        self._downstream = {}; self._ext_in = {}; self._ext_out = {}
        for sblk, sport, dblk, dport in self.connections:
            if sblk == _EXT:
                self._ext_in.setdefault(sport, []).append((dblk, dport))  # nombre externo → destinos
            elif dblk == _EXT:
                self._ext_out[dport] = (sblk, sport)
            else:
                self._downstream.setdefault((sblk, sport), []).append((dblk, dport))

        self._built = True
        return self

    # ---------------- estados gestionados por el modelo ----------------
    def initialize(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        initial_states: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        strict_validate: bool = True,
        auto_build: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if not self._built:
            if auto_build: self.build()
            else: raise RuntimeError("Model.build() debe llamarse antes de initialize().")

        # Mover buffers/parámetros de todo el modelo al device/dtype solicitados
        if device is not None or dtype is not None:
            super().to(device=device, dtype=dtype)

        states: Dict[str, torch.Tensor] = {}
        for name, blk in self.blocks.items():
            for pname in blk._param_attr.keys():
                blk.expand_param(pname, batch_size)
            st = blk.init_state(batch_size, device=device, dtype=dtype)
            if initial_states and name in initial_states and st.shape[1] > 0:
                spec = initial_states[name]
                for key, val in spec.items():
                    if isinstance(key, (int, slice)):
                        sl = key
                    else:
                        if key not in blk.state_alias:
                            raise KeyError(f"{name}: estado '{key}' no existe en state_alias.")
                        sl = blk.state_alias[key]
                    t = torch.as_tensor(val, device=st.device, dtype=st.dtype)
                    if t.ndim == 1: t = t.unsqueeze(1)
                    if isinstance(sl, int): sl = slice(sl, sl+1)
                    if strict_validate:
                        if t.ndim != 2 or t.shape[0] != st.shape[0]:
                            raise ValueError(f"{name}: estado '{key}' debe ser (B,k); recibido {tuple(t.shape)}.")
                        k_slice = (sl.stop - sl.start)
                        if t.shape[1] != k_slice:
                            raise ValueError(f"{name}: ancho '{key}'={t.shape[1]} != slice {k_slice}.")
                    st[:, sl] = t
            states[name] = st

        # Guardar dentro del modelo
        self._states = states
        any_st = next(iter(states.values()))
        self._B = int(any_st.shape[0]); self._device = any_st.device; self._dtype = any_st.dtype
        return states  # se mantiene retorno por compatibilidad

    @property
    def has_states(self) -> bool:
        return self._states is not None

    @property
    def states(self) -> Dict[str, torch.Tensor]:
        if self._states is None:
            raise RuntimeError("Model: no hay estados. Llama a initialize() primero.")
        return self._states

    def set_states(self, states: Dict[str, torch.Tensor], *, validate_shapes: bool = True):
        if not isinstance(states, dict) or not states:
            raise ValueError("Model.set_states: 'states' debe ser dict no vacío {block: Tensor(B,S)}.")
        if validate_shapes:
            ref = next(iter(states.values()))
            B = int(ref.shape[0]); device = ref.device; dtype = ref.dtype
            for n, st in states.items():
                if n not in self.blocks:
                    raise KeyError(f"Model.set_states: bloque desconocido '{n}'.")
                if st.ndim != 2 or st.shape[0] != B:
                    raise ValueError(f"{n}: state debe ser (B,S). Recibido {tuple(st.shape)}.")
                if st.device != device or st.dtype != dtype:
                    raise ValueError(f"{n}: device/dtype inconsistentes.")
            self._B, self._device, self._dtype = B, device, dtype
        self._states = states

    def snapshot_states(self, *, clone: bool = True, detach: bool = False) -> Dict[str, torch.Tensor]:
        if not self.has_states:
            raise RuntimeError("Model.snapshot_states: no hay estados.")
        snap: Dict[str, torch.Tensor] = {}
        for n, st in self.states.items():
            t = st
            if detach: t = t.detach()
            if clone:  t = t.clone()
            snap[n] = t
        return snap

    def restore_states(self, snapshot: Dict[str, torch.Tensor]):
        self.set_states(snapshot, validate_shapes=True)


    # ---------------- DAE: registro de restricciones y residual global ----------------
    def add_constraint_eq(self, name: str, fn):
        """Registra una restricción de igualdad h(t,x,z,u,params)=0 (Tensor (B,r))."""
        if name in self._eq_constraints:
            raise KeyError(f"Constraint '{name}' ya existe.")
        self._eq_constraints[name] = fn
        return self

    def add_complementarity(self, name: str, c_fn, lambda_name: Optional[str] = None):
        """Registra una desigualdad como complementariedad 0 <= c ⟂ λ >= 0.
        Nota: la resolución se hace en el solver; aquí solo registramos.
        """
        if name in self._compl_constraints:
            raise KeyError(f"Complementarity '{name}' ya existe.")
        self._compl_constraints[name] = (c_fn, lambda_name or f"lambda_{name}")
        return self

    def build_residual(
        self,
        t: float,
        x_all: Dict[str, torch.Tensor],
        u_all: Dict[str, Dict[str, torch.Tensor]],
        z: Optional[torch.Tensor] = None,
        params_all: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Concatena residuales algebraicos globales:
         (1) Residuales locales de bloques (algebraic_residual())
         (2) Restricciones de igualdad registradas con add_constraint_eq()
        MVP: el residual de cierre de lazos (SCC) se incorporará en PR posterior.
        Devuelve Tensor (B, R) o (B,0) si no hay residuales.
        """
        B = None; device = None; dtype = None
        for st in x_all.values():
            B = st.shape[0]; device = st.device; dtype = st.dtype; break
        if B is None:
            for m in u_all.values():
                for ten in m.values():
                    B = ten.shape[0]; device = ten.device; dtype = ten.dtype; break
                if B is not None: break
        if B is None:
            B, device, dtype = 1, torch.device("cpu"), torch.float32

        chunks = []

        # (1) residuales algebraicos locales por bloque
        for bname, blk in self.blocks.items():
            xi = x_all.get(bname, None)
            ui = u_all.get(bname, {})
            gres = blk.algebraic_residual(t, xi, ui, getattr(blk, "params", {}))
            if isinstance(gres, dict):
                for ten in gres.values():
                    if ten is not None and ten.numel() > 0:
                        chunks.append(ten.reshape(B, -1))
            elif isinstance(gres, torch.Tensor) and gres.numel() > 0:
                chunks.append(gres.reshape(B, -1))

        # (2) restricciones de igualdad globales
        for _n, fn in self._eq_constraints.items():
            h = fn(t, x_all, z, u_all, params_all or {})
            if isinstance(h, torch.Tensor) and h.numel() > 0:
                chunks.append(h.reshape(B, -1))

        if not chunks:
            return torch.zeros((B, 0), device=device, dtype=dtype)
        return torch.cat(chunks, dim=-1)

    # ---------------- utilidades ----------------
    def required_externals(self) -> List[str]:
        return sorted(self._ext_in.keys())

    def output_names(self) -> List[str]:
        return sorted(self._ext_out.keys())
