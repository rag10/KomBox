# kombox/core/utils.py
from __future__ import annotations

import os
import re
import glob
from typing import Dict, List, Tuple, Optional, Iterable, Union

import numpy as np
import math
import torch


def list_npz_parts(path_pattern: str) -> List[str]:
    """
    Enumera y ordena los ficheros .npz que corresponden a un patrón de chunks.

    Soporta:
      - Patrón con {part}:  "runs/msd_trace_part{part:04d}.npz"
      - Patrón tipo glob:   "runs/msd_trace_part*.npz"

    Devuelve:
      Lista de rutas ordenadas por el índice 'part' (o por orden natural si glob).
    """
    parts: List[Tuple[int, str]] = []

    if "{part" in path_pattern and "}" in path_pattern:
        # construimos regex a partir del patrón
        # Ej: "a{part:04d}b" -> prefix="a", suffix="b"
        m = re.search(r"\{part[^}]*\}", path_pattern)
        assert m is not None
        prefix = path_pattern[: m.start()]
        suffix = path_pattern[m.end():]

        # listamos todos los ficheros en el directorio del patrón y filtramos por prefijo/sufijo
        folder = os.path.dirname(prefix) or "."
        cand = glob.glob(os.path.join(folder, "*"))
        rx = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix))
        for p in cand:
            mo = rx.fullmatch(p)
            if mo:
                idx = int(mo.group(1))
                parts.append((idx, p))
        parts.sort(key=lambda x: x[0])
        return [p for _, p in parts]

    # fallback: tratar como glob
    files = glob.glob(path_pattern)
    # ordenar por número al final si existe, si no lexicográfico
    def _nat_key(s: str):
        m = re.search(r"(\d+)(?!.*\d)", s)
        return (int(m.group(1)) if m else float("inf"), s)
    files.sort(key=_nat_key)
    return files


def _concat_arrays(arrs: List[np.ndarray], axis: int = 0) -> np.ndarray:
    if not arrs:
        raise ValueError("No hay arrays para concatenar.")
    # Verificar compatibilidad (todas las dims salvo axis)
    ref = arrs[0].shape
    for a in arrs[1:]:
        if len(a.shape) != len(ref):
            raise ValueError(f"Rank inconsistente al concatenar: {a.shape} vs {ref}")
        for i, (d1, d2) in enumerate(zip(a.shape, ref)):
            if i == axis:
                continue
            if d1 != d2:
                raise ValueError(f"Dim incompatible en eje {i}: {d1} vs {d2}")
    return np.concatenate(arrs, axis=axis)


def load_npz_series(
    path_pattern: str,
    *,
    strip_prefix: bool = True,
    to_torch: bool = False,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> Dict[str, Union[np.ndarray, "torch.Tensor"]]:
    """
    Carga todos los chunks .npz que coincidan con `path_pattern` y concatena
    cada clave a lo largo del eje temporal (0). Devuelve un solo diccionario.

    Args:
      path_pattern: patrón con {part} o glob que apunte a los .npz de NPZChunkRecorder.
      strip_prefix: si True, elimina 'outs/' y 'state/' de las claves resultantes.
      to_torch:     si True, convierte los arrays a torch.Tensor.
      device, dtype: opcionalmente fija destino de los tensores.

    Returns:
      Dict[str, array] con:
        - "t": (T_total,) si estaba presente
        - "outs/<blk>.<port>": (T_total, B, n)  (o "<blk>.<port>" si strip_prefix)
        - "state/<blk>.state:<alias|i|slice>": (T_total, B, k)
    """
    files = list_npz_parts(path_pattern)
    if not files:
        raise FileNotFoundError(f"No se encontraron .npz para patrón: {path_pattern}")

    # Leer todas las claves y acumular por clave
    buckets: Dict[str, List[np.ndarray]] = {}
    for fp in files:
        with np.load(fp, allow_pickle=False) as z:
            for key in z.files:
                buckets.setdefault(key, []).append(z[key])

    # Concatenar por clave
    out_np: Dict[str, np.ndarray] = {}
    for key, arrs in buckets.items():
        # t es 1D; el resto (N,B,...) concatenamos en eje 0
        if key == "t":
            vecs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrs]
            out_np[key] = np.concatenate(vecs, axis=0)
        else:
            out_np[key] = _concat_arrays([np.asarray(a) for a in arrs], axis=0)

    # Renombrar claves (opcional)
    if strip_prefix:
        renamed: Dict[str, np.ndarray] = {}
        for k, v in out_np.items():
            if k.startswith("outs/"):
                renamed[k[len("outs/"):]] = v
            elif k.startswith("state/"):
                renamed[k[len("state/"):]] = v
            else:
                renamed[k] = v
        out_np = renamed

    # Convertir a torch si se pide
    if to_torch:
        if torch is None:
            raise RuntimeError("Torch no está disponible; instala torch o usa to_torch=False.")
        out_t: Dict[str, "torch.Tensor"] = {}
        for k, v in out_np.items():
            t = torch.from_numpy(v)
            if dtype is not None:
                t = t.to(dtype=dtype)
            if device is not None:
                t = t.to(device=device)
            out_t[k] = t
        return out_t

    return out_np


def _resolve_block_and_param(model, param_path: str) -> Tuple[str, str]:
    """'spring.k' -> ('spring','k'). Solo tope de jerarquía por ahora."""
    if "." not in param_path:
        raise ValueError(f"Ruta de parámetro inválida '{param_path}'. Usa 'bloque.param'.")
    blk_name, p_name = param_path.split(".", 1)
    if blk_name not in model.blocks:
        raise KeyError(f"Bloque '{blk_name}' no existe en el modelo.")
    blk = model.blocks[blk_name]
    if p_name not in blk._param_attr:
        raise KeyError(f"Parámetro '{p_name}' no declarado en bloque '{blk_name}'.")
    return blk_name, p_name

def _param_width(model, blk_name: str, p_name: str) -> int:
    blk = model.blocks[blk_name]
    return int(blk._param_width[p_name])

def _to_row_vector(value: Any, D: int, device=None, dtype=None) -> torch.Tensor:
    """
    Convierte un valor a vector (D,). Reglas:
      - escalar -> repite a (D,)
      - secuencia/1D -> debe tener len==D
      - tensor 2D (1,D) -> squeeze
    """
    t = torch.as_tensor(value, device=device, dtype=dtype)
    if t.ndim == 0:
        return t.expand(D) if D > 1 else t.reshape(1)
    if t.ndim == 1:
        if t.shape[0] != D:
            raise ValueError(f"Valor de parámetro de longitud {t.shape[0]} no coincide con D={D}.")
        return t
    if t.ndim == 2 and t.shape[0] == 1 and t.shape[1] == D:
        return t.squeeze(0)
    raise ValueError(f"Valor de parámetro con shape {tuple(t.shape)} no soportado; se espera (D,) para D={D}.")

def apply_parameter_grid(
    model,
    grid: Dict[str, Sequence[Any]],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """
    Aplica una malla cartesiana de parámetros en batch.
    Args:
      model: Model ya construido (build hecho). No es necesario initialize aún.
      grid: dict {'bloque.param': [v1, v2, ...], ...}
    Returns:
      B = producto de longitudes -> tamaño de batch resultante.
    Efectos:
      para cada 'bloque.param' fija el tensor 2D (B, D) en el Block correspondiente.
    """
    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid debe ser un dict no vacío {'bloque.param': [valores,...]}")

    # Orden determinista: respetar orden de inserción del dict
    keys: List[str] = list(grid.keys())
    lengths: List[int] = [len(grid[k]) for k in keys]
    if any(L == 0 for L in lengths):
        raise ValueError("Todas las listas de valores deben tener al menos 1 elemento.")
    B = 1
    for L in lengths: B *= int(L)

    # Resolver device/dtype por defecto
    if device is None or dtype is None:
        try:
            p0 = next(model.parameters())
            if device is None: device = p0.device
            if dtype is None: dtype = p0.dtype
        except StopIteration:
            device = device or torch.device("cpu")
            dtype  = dtype  or torch.float32

    # Pre-calcular anchos D por parámetro
    widths: Dict[str, int] = {}
    for path in keys:
        blk_name, p_name = _resolve_block_and_param(model, path)
        widths[path] = _param_width(model, blk_name, p_name)

    # Para cada parámetro construir su matriz (B, D)
    batched_values: Dict[str, torch.Tensor] = {}
    # patrón de repetición cartesiano
    # para key_i: repeat_each = prod_{j>i} Lj ; repeat_block = prod_{j<i} Lj
    prod_suffix = [1]*(len(keys)+1)
    for i in range(len(keys)-1, -1, -1):
        prod_suffix[i] = prod_suffix[i+1] * int(lengths[i])
    prod_prefix = [1]*(len(keys)+1)
    for i in range(1, len(keys)+1):
        prod_prefix[i] = prod_prefix[i-1] * int(lengths[i-1])

    for i, path in enumerate(keys):
        vals = grid[path]
        Li = int(lengths[i])
        D  = int(widths[path])

        row_stack = torch.stack([_to_row_vector(v, D, device=device, dtype=dtype) for v in vals], dim=0)  # (Li, D)
        repeat_each  = prod_suffix[i+1]  # cuántas veces se repite cada valor consecutivamente
        repeat_block = prod_prefix[i]    # cuántas veces se repite la secuencia completa Li*repeat_each

        # (Li, D) -> (Li*repeat_each, D): repite cada fila consecutivamente
        expanded = row_stack.repeat_interleave(repeat_each, dim=0)
        # -> (repeat_block*Li*repeat_each, D) == (B, D)
        full = expanded.repeat((repeat_block, 1))
        assert full.shape[0] == B and full.shape[1] == D
        batched_values[path] = full

    # Asignar en los bloques
    for path, mat in batched_values.items():
        blk_name, p_name = _resolve_block_and_param(model, path)
        blk = model.blocks[blk_name]
        blk.set_param(p_name, mat)  # acepta 2D (B,D)
    return B

# --- Barridos (sweeps) y utilidades de grid ----------------------------------
from typing import Dict, Sequence, Any, Optional, List
import math

def sweep_lin(path: str, start: float, stop: float, num: int, *, endpoint: bool = True) -> Dict[str, List[float]]:
    """
    Crea un grid unidimensional en escala lineal para 'path' (p.ej. 'spring.k').
    Devuelve: {'spring.k': [v1, v2, ...]}
    """
    if num <= 0:
        raise ValueError("num debe ser > 0")
    if num == 1:
        vals = [float(stop)]
    else:
        step = (stop - start) / (num - (1 if endpoint else 0))
        vals = [float(start + i * step) for i in range(num)]
        if endpoint and num > 1:
            vals[-1] = float(stop)
    return {path: vals}

def sweep_log(path: str, start_exp: float, stop_exp: float, num: int, *, base: float = 10.0, endpoint: bool = True) -> Dict[str, List[float]]:
    """
    Crea un grid unidimensional en escala logarítmica: base**exp, exp en [start_exp, stop_exp].
    Ej.: sweep_log('damper.c', -2, 1, 4) -> [1e-2, 1e-1, 1e0, 1e1]
    """
    if base <= 0 or base == 1.0:
        raise ValueError("base debe ser >0 y != 1")
    g = sweep_lin(path, start_exp, stop_exp, num, endpoint=endpoint)[path]
    vals = [float(base ** e) for e in g]
    return {path: vals}

def sweep_list(path: str, values: Sequence[Any]) -> Dict[str, List[Any]]:
    """
    Crea un grid unidimensional con una lista arbitraria de valores (numérica o vector/tensor).
    """
    if not isinstance(values, Sequence) or len(values) == 0:
        raise ValueError("values debe ser una secuencia no vacía")
    return {path: list(values)}

def combine_grids(*grids: Dict[str, Sequence[Any]], allow_overwrite: bool = False) -> Dict[str, List[Any]]:
    """
    Fusiona varios grids {'path': [vals]} en uno. Por defecto no permite paths duplicados.
    """
    out: Dict[str, List[Any]] = {}
    for g in grids:
        for k, v in g.items():
            if k in out and not allow_overwrite:
                raise KeyError(f"Parámetro duplicado en combine_grids: '{k}'. Usa allow_overwrite=True para sobrescribir.")
            out[k] = list(v)
    return out
