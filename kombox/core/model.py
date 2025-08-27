# kombox/core/model.py
from __future__ import annotations
from typing import Dict, Mapping, Tuple, Optional, List
from collections import deque
import inspect
import torch
import torch.nn as nn

from .block import Block
from .topology import FeedthroughGraph

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

        self._edges = []                 # lista de (src_blk, src_port, dst_blk, dst_port)
        self._constraints = []           # lista de (name, fn)
        self.topology = type("Topo", (), {"sccs": []})()

        self._has_pure_algebraic_cycle = False
        self._algebraic_cycles = []  # lista de ciclos detectados (por nombres de bloque)
        self._z_dim = 0
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
        self._edges.append((sblk, sport, dblk, dport))
        return self

    def connect_from_strings(self, *arrows: str):
        for arrow in arrows:
            left, right = arrow.split("->")
            self.connect(left.strip(), right.strip())
        return self
    
    # ---------------- azúcar para externals ----------------
    def bind_external(self, name: str, to):
        """
        Conecta una entrada externa 'name' hacia uno o varios destinos 'blk.port'.
        Equivalente a: connect(name, "blk.port")
        """
        if isinstance(to, (list, tuple)):
            for dst in to:
                self.connect(name, dst)
        else:
            self.connect(name, to)
        return self

    def expose(self, name: str, from_):
        """
        Expone como salida externa 'name' una salida 'blk.port' interna (o varias).
        Equivalente a: connect("blk.port", name)
        """
        if isinstance(from_, (list, tuple)):
            for src in from_:
                self.connect(src, name)
        else:
            self.connect(from_, name)
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
        if not blk.out_specs or not hasattr(blk, "_expose_outputs"):
            self._feedthrough_cache[name] = False
            return False
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

    def _toposort_or_cycle(self, nodes, edges):
        adj = {u: [] for u in nodes}
        for u, v in edges:
            adj[u].append(v)
        color = {u: 0 for u in nodes}  # 0=white,1=gray,2=black
        stack = []
        order = []

        def dfs(u: str):
            color[u] = 1; stack.append(u)
            for v in adj[u]:
                if color[v] == 0:
                    if dfs(v): 
                        return True
                elif color[v] == 1:
                    # ciclo detectado
                    cyc = [v]; i = len(stack) - 1
                    while i >= 0 and stack[i] != v:
                        cyc.append(stack[i]); i -= 1
                    cyc.append(v); cyc.reverse()
                    # En vez de lanzar, marcamos y guardamos
                    self._has_pure_algebraic_cycle = True
                    self._algebraic_cycles.append(cyc)
                    # devolvemos True para cortar búsqueda aquí,
                    # pero NO lanzamos excepción
                    return True
            color[u] = 2; order.append(u); stack.pop()
            return False

        for u in nodes:
            if color[u] == 0:
                if dfs(u):
                    # Hemos visto un ciclo. Devolvemos un orden parcial (vacío) para indicar
                    # que hay lazos algebraicos; build continuará y podremos usar SCCs.
                    return []
        order.reverse()
        return order

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
        """Registra una restricción/ecuación global g(...)=0.
        fn: callable(t, states, inbuf, model[, z]) -> Tensor (B, m)
        """
        self._constraints.append((name, fn))
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
        states: Optional[Dict[str, torch.Tensor]] = None,
        inbuf: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
        z: Optional[torch.Tensor] = None,
        params_all: Optional[Dict[str, dict]] = None,
    ):
        """Concatena residuales algebraicos de bloques + restricciones globales.
        Retorna: (Tensor (B, R_total), detail: dict por-bloque+global).

        - NO requiere haber llamado a `initialize()` si `states` es None (soporta casos puramente algebraicos).
        - Acepta `z` por compatibilidad con solvers implícitos.
        """
        # Nunca acceder a self.states si el caller no lo pasó
        states_in: Dict[str, torch.Tensor] = {} if states is None else states
        # Si inbuf no viene, poner dict vacío por bloque
        if inbuf is None:
            inbuf = {n: {} for n in self.blocks.keys()}

        pieces: List[torch.Tensor] = []
        detail: Dict[str, torch.Tensor] = {}

        def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
            return tensor if tensor.ndim == 2 else tensor.view(tensor.shape[0], -1)

        # Inferir B: primero desde estados_in, si no, desde inbuf, si no, B=1
        B = None
        for v in states_in.values():
            if isinstance(v, torch.Tensor) and v.numel() > 0:
                B = v.shape[0]
                break
        if B is None:
            for bd in inbuf.values():
                for ten in bd.values():
                    if isinstance(ten, torch.Tensor) and ten.numel() > 0:
                        B = ten.shape[0]
                        break
                if B is not None:
                    break
        if B is None:
            B = 1

        # Residuales locales por bloque
        for bname, blk in self.blocks.items():
            state = states_in.get(bname, torch.empty((B, 0), dtype=torch.float32))
            inputs = inbuf.get(bname, {})
            gres = blk.algebraic_residual(t, state, inputs, getattr(blk, "params", None))

            if isinstance(gres, dict):
                if len(gres) > 0:
                    cols = []
                    for v in gres.values():
                        v = _ensure_2d(v)
                        if v.shape[0] == 1 and B > 1:
                            v = v.expand(B, -1)           # <<< expandimos batch 1 → B
                        elif v.shape[0] not in (1, B):
                            raise ValueError(
                                f"Residual local de '{bname}' con batch={v.shape[0]} incompatible con B={B}"
                            )
                        cols.append(v)
                    gcat = torch.cat(cols, dim=1)
                    pieces.append(gcat); detail[bname] = gcat

            elif hasattr(gres, "ndim"):
                if gres.numel() > 0:
                    gres = _ensure_2d(gres)
                    if gres.shape[0] == 1 and B > 1:
                        gres = gres.expand(B, -1)         # <<< expandimos batch 1 → B
                    elif gres.shape[0] not in (1, B):
                        raise ValueError(
                            f"Residual local de '{bname}' con batch={gres.shape[0]} incompatible con B={B}"
                        )
                    pieces.append(gres); detail[bname] = gres

        # Restricciones globales (acepta 4 o 5 argumentos)
        for name, fn in self._constraints:
            g = None
            try:
                sig = inspect.signature(fn)
                if len(sig.parameters) >= 5:
                    g = fn(t, states_in, inbuf, self, z)
                else:
                    g = fn(t, states_in, inbuf, self)
            except TypeError:
                try:
                    g = fn(t, states_in, inbuf)
                except TypeError as e:
                    raise TypeError(f"Constraint '{name}' no acepta firma compatible") from e

            if g is None:
                continue
            g = _ensure_2d(g)
            if g.shape[0] == 1 and B > 1:
                g = g.expand(B, -1)                       # <<< expandimos batch 1 → B
            elif g.shape[0] not in (1, B):
                raise ValueError(
                    f"Constraint '{name}' con batch={g.shape[0]} incompatible con B={B}"
                )

            pieces.append(g); detail[f"__global__::{name}"] = g

        if len(pieces) == 0:
            return torch.zeros((B, 0), dtype=torch.float32), detail
        return torch.cat(pieces, dim=1), detail

    def declare_algebraic_dim(self, dim: int):
        """Declara la dimensión global de las variables algebraicas z (B, dim)."""
        if dim < 0:
            raise ValueError("Algebraic dim must be >= 0")
        self._z_dim = int(dim)
        return self

    def analyze_topology(self):
        g = FeedthroughGraph()
        for bname in self.blocks.keys():
            g.add_node(bname)
        for (sb, sp, db, dp) in getattr(self, "_edges", []):
            g.add_edge(sb, db)
        self.topology.sccs = g.sccs()
        return self.topology.sccs

    # ---------------- utilidades ----------------
    def required_externals(self) -> List[str]:
        return sorted(self._ext_in.keys())

    def output_names(self) -> List[str]:
        return sorted(self._ext_out.keys())
