# kombox/core/topology.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Set

class FeedthroughGraph:
    """Grafo dirigido entre nombres de bloques para análisis algebraico/instantáneo."""
    def __init__(self):
        self.adj: Dict[str, List[str]] = {}
        self.nodes: Set[str] = set()

    def add_edge(self, src: str, dst: str):
        self.nodes.add(src); self.nodes.add(dst)
        self.adj.setdefault(src, []).append(dst)
        self.adj.setdefault(dst, [])  # asegurar clave

    def add_node(self, n: str):
        self.nodes.add(n)
        self.adj.setdefault(n, [])

    # ---- Tarjan SCC ----
    def sccs(self) -> List[List[str]]:
        index = 0
        stack: List[str] = []
        onstack: Set[str] = set()
        idx: Dict[str, int] = {}
        low: Dict[str, int] = {}
        out: List[List[str]] = []

        def strongconnect(v: str):
            nonlocal index
            idx[v] = index; low[v] = index; index += 1
            stack.append(v); onstack.add(v)

            for w in self.adj.get(v, []):
                if w not in idx:
                    strongconnect(w)
                    low[v] = min(low[v], low[w])
                elif w in onstack:
                    low[v] = min(low[v], idx[w])

            if low[v] == idx[v]:
                comp = []
                while True:
                    w = stack.pop()
                    onstack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)

        for v in list(self.nodes):
            if v not in idx:
                strongconnect(v)
        return out
