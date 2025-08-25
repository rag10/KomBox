# kombox/vis/graph.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import warnings

import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from kombox.core.block import Block
from kombox.core.model import Model


# ----------------------------- Helpers internos ------------------------------

def _port_names(block: Block, direction: str) -> List[str]:
    if direction == "in":
        return sorted(list(block.in_specs.keys()))
    elif direction == "out":
        return sorted(list(block.out_specs.keys()))
    raise ValueError("direction debe ser 'in' o 'out'")


def _state_names(block: Block) -> List[str]:
    if getattr(block, "state_alias", None):
        # ordenar por índice/slice inicial para que salga legible
        def _start(a): 
            sl = block.state_alias[a]
            if isinstance(sl, slice): return sl.start if sl.start is not None else 0
            return int(sl)
        return sorted(block.state_alias.keys(), key=_start)
    # alias no definidos: usar índices [0..S-1]
    S = block.state_size()
    return [f"s{i}" for i in range(S)]


def _maybe_states_shape(model: Optional[Model], block_name: str) -> Optional[Tuple[int, int]]:
    if model is None:
        return None
    if not hasattr(model, "states"):
        return None
    st = model.states.get(block_name, None)
    if st is None:
        return None
    return tuple(st.shape)


# -------------------------- Impresión en consola -----------------------------

def _maybe_states_shape(model, block_name: str):
    st_dict = getattr(model, "_states", None)  # dict o None
    if not isinstance(st_dict, dict):
        return None
    st = st_dict.get(block_name, None)
    if st is None:
        return None
    return tuple(st.shape)


def print_block_details(block_name: str, block, model=None, level: int = 0):
    indent = "\t" * level
    print(indent + f"● {block_name} : {block.__class__.__name__}")
    print(indent + "-" * (len(block_name) + 5))

    # IO
    ins = sorted(list(block.in_specs.keys()))
    outs = sorted(list(block.out_specs.keys()))
    print(indent + f"\t- Inputs:  [{', '.join(repr(n) for n in ins)}]")
    print(indent + f"\t- Outputs: [{', '.join(repr(n) for n in outs)}]")

    # Estados
    S = block.state_size()
    st_shape = _maybe_states_shape(model, block_name) if model is not None else None
    if S == 0:
        print(indent + "\t- States:  (none)")
    else:
        shape_txt = f", shape={st_shape}" if st_shape is not None else ""
        print(indent + f"\t- States (S={S}{shape_txt}):")
        # nombres de estado si hay alias; si no, índices
        if getattr(block, "state_alias", None):
            # ordenar por posición
            def _start(a):
                sl = block.state_alias[a]
                if isinstance(sl, slice):
                    return sl.start if sl.start is not None else 0
                return int(sl)
            names = sorted(block.state_alias.keys(), key=_start)
        else:
            names = [f"s{i}" for i in range(S)]
        for n in names:
            alias = block.state_alias.get(n, None) if getattr(block, "state_alias", None) else None
            print(indent + f"\t   ↳ {n}: {alias}")

    # Parámetros (resumen)
    if hasattr(block, "_param_attr") and block._param_attr:
        params = list(block._param_attr.keys())
        print(indent + "\t- Params:", ", ".join(params))
    else:
        print(indent + "\t- Params: (none)")
    print()


def print_model_overview(model):
    print(f"Modelo: {model.name}")
    print("=" * (8 + len(model.name)))
    # Si no está inicializado, no intentes leer shapes
    initialized = isinstance(getattr(model, "_states", None), dict)
    if not initialized:
        print("(nota) El modelo aún no está inicializado: se muestran puertos y tamaños S, "
              "pero no 'shape' de estados.")
    for name, blk in model.blocks.items():
        print_block_details(name, blk, model=model if initialized else None, level=1)



# ----------------------------- Visualización --------------------------------

def visualize_model(model: Model, include_io: bool = True, layout: str = "spring",
                    figsize: Tuple[int, int] = (11, 9), seed: int = 42):
    """
    Visualiza el grafo del modelo (nodos = bloques). Opcionalmente muestra:
      - “burbujas” laterales con los puertos (in/out) y debajo los estados.
      - nodos extra para entradas/salidas externas.

    Parameters
    ----------
    model : Model KomBox
    include_io : bool
        Si True, dibuja burbujas de entradas/salidas/estados.
    layout : {'spring','kamada','circular','planar'}
    figsize : (w,h)
    seed : int

    Returns
    -------
    fig, ax : Figure, Axes
    """
    G = nx.DiGraph()

    # Nodos principales (bloques del modelo)
    node_shapes: Dict[str, str] = {}
    node_labels: Dict[str, str] = {}
    node_colors: Dict[str, str] = {}

    for blk_name, blk in model.blocks.items():
        G.add_node(blk_name)
        node_shapes[blk_name] = "s"              # cuadrados para bloques
        node_labels[blk_name] = blk_name
        node_colors[blk_name] = "#BBD7FF"        # azul claro

    edge_labels: Dict[Tuple[str, str], List[str]] = {}

    # Conexiones internas (usamos _downstream)
    for (src_blk, src_port), dst_list in model._downstream.items():
        for (dst_blk, dst_port) in dst_list:
            G.add_edge(src_blk, dst_blk)
            lab = f"{src_blk}.{src_port} → {dst_blk}.{dst_port}"
            edge_labels.setdefault((src_blk, dst_blk), []).append(lab)

    # Entradas externas (nodos verdes que apuntan a destinos)
    ext_in_nodes: List[str] = []
    if hasattr(model, "_ext_in") and model._ext_in:
        for ext_name, targets in model._ext_in.items():
            io_node = f"[in] {ext_name}"
            ext_in_nodes.append(io_node)
            G.add_node(io_node)
            node_shapes[io_node] = "o"
            node_labels[io_node] = ext_name
            node_colors[io_node] = "#6BD66B"  # verde
            for (dst_blk, dst_port) in targets:
                G.add_edge(io_node, dst_blk)
                lab = f"{ext_name} → {dst_blk}.{dst_port}"
                edge_labels.setdefault((io_node, dst_blk), []).append(lab)

    # Salidas externas (nodos naranjas desde orígenes)
    ext_out_nodes: List[str] = []
    if hasattr(model, "_ext_out") and model._ext_out:
        for ext_name, sources in model._ext_out.items():
            if isinstance(sources, tuple):
                sources = [sources]
            io_node = f"[out] {ext_name}"
            ext_out_nodes.append(io_node)
            G.add_node(io_node)
            node_shapes[io_node] = "o"
            node_labels[io_node] = ext_name
            node_colors[io_node] = "#FFB066"  # naranja
            for (src_blk, src_port) in sources:
                G.add_edge(src_blk, io_node)
                lab = f"{src_blk}.{src_port} → {ext_name}"
                edge_labels.setdefault((src_blk, io_node), []).append(lab)

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "planar":
        try:
            pos = nx.planar_layout(G)
        except Exception:
            warnings.warn("planar_layout ha fallado; usando spring_layout.")
            pos = nx.spring_layout(G, seed=seed)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # Dibujo
    fig, ax = plt.subplots(figsize=figsize)
    SQUARE_SIZE = 3000
    CIRCLE_SIZE = 1100

    # Dibuja por forma
    for shape in {"s", "o"}:
        nodes = [n for n in G.nodes if node_shapes.get(n, "s") == shape]
        if not nodes:
            continue
        colors = [node_colors[n] for n in nodes]
        sizes = [SQUARE_SIZE if shape == "s" else CIRCLE_SIZE] * len(nodes)
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_color=colors, node_shape=shape,
            node_size=sizes, ax=ax, linewidths=1.0, edgecolors="#444"
        )

    # Aristas (con ligera curvatura)
    edge_kwargs = dict(
        arrowstyle="-|>", arrowsize=18, edge_color="#666",
        connectionstyle="arc3,rad=0.12"
    )
    try:
        edge_kwargs["min_source_margin"] = 12
        edge_kwargs["min_target_margin"] = 12
    except Exception:
        pass

    edge_artists = nx.draw_networkx_edges(G, pos, ax=ax, **edge_kwargs)
    if edge_artists is not None:
        try:
            for a in edge_artists: a.set_zorder(3.0)
        except TypeError:
            edge_artists.set_zorder(3.0)

    # Etiquetas de nodos
    for n, (x, y) in pos.items():
        ax.text(
            x, y + 0.00, node_labels.get(n, n),
            ha="center", va="bottom", fontsize=12, color="#222",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.6),
            zorder=4.0
        )

    # Etiquetas de aristas (compuestas si hay varias)
    for (src, dst), labels in edge_labels.items():
        src_pos = np.array(pos[src]); dst_pos = np.array(pos[dst])
        direction = dst_pos - src_pos
        midpoint = src_pos + 0.65 * direction
        perp = np.array([-direction[1], direction[0]])
        norm = np.linalg.norm(perp); perp = perp / norm * 0.008 if norm > 0 else np.array([0.0, 0.0])
        lbl = "\n".join(labels)
        ax.text(
            *(midpoint - perp), lbl,
            fontsize=8, ha="center", va="center", color="#0A58CA",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            zorder=3.5
        )

    # Burbujas IO + estados junto a cada bloque
    if include_io:
        bubble_dx_out, bubble_dx_in = 0.10, 0.10
        start_dx_out, start_dx_in = 0.05, 0.05
        bubble_dy = 0.04
        line_w = 1.2

        def _draw_bubble(x, y, color_face, color_edge, s=100):
            ax.scatter(x, y, s=s, color=color_face, edgecolors=color_edge, zorder=3.2)

        for blk_name, blk in model.blocks.items():
            base_x, base_y = pos[blk_name]

            # OUT bubbles (derecha)
            for i, pname in enumerate(_port_names(blk, "out")):
                ox = base_x + bubble_dx_out
                oy = base_y - bubble_dy * i
                sx = base_x + start_dx_out
                sy = base_y
                ax.plot([sx, ox], [sy, oy], color="#D2691E", linewidth=line_w, alpha=0.9, zorder=2.8)
                _draw_bubble(ox, oy, "#FFB066", "#D2691E")
                ax.text(ox + 0.02, oy, pname, fontsize=7, color="#D2691E", va="center", zorder=3.4)

            # IN bubbles (izquierda)
            for i, pname in enumerate(_port_names(blk, "in")):
                ix = base_x - bubble_dx_in
                iy = base_y - bubble_dy * i
                tx = base_x - start_dx_in
                ty = base_y
                ax.plot([ix, tx], [iy, ty], color="#2E8B57", linewidth=line_w, alpha=0.9, zorder=2.8)
                _draw_bubble(ix, iy, "#6BD66B", "#2E8B57")
                ax.text(ix - 0.02, iy, pname, fontsize=7, color="#2E8B57", va="center", ha="right", zorder=3.4)

            # STATES bubbles (debajo)
            st_names = _state_names(blk)
            if st_names:
                state_offset_down = 0.055
                bubble_dy_state = 0.045
                for i, sname in enumerate(st_names):
                    sx = base_x
                    sy = base_y - (state_offset_down + i * bubble_dy_state)
                    ax.plot([base_x, sx], [base_y, sy], color="#1F4EAD", linewidth=line_w, alpha=0.9, zorder=2.8)
                    _draw_bubble(sx, sy, "#8FB8FF", "#1F4EAD")
                    ax.text(sx + 0.015, sy, sname, fontsize=7, color="#1F4EAD", va="center", ha="left", zorder=3.4)

    ax.axis("off")
    ax.set_title(f"Model '{model.name}'", fontsize=13)
    plt.tight_layout()
    plt.draw()
    return fig, ax
