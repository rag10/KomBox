# kombox/core/recorders.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Mapping, Union
import os
import math
import numpy as np
import torch

from .model import Model
from .block import Block


AliasKey = Union[str, int, slice]


def _ensure_2d_by_index(t: torch.Tensor, key: AliasKey) -> torch.Tensor:
    """
    Asegura que la selección de estado sea 2D (B,k).
    - int -> slice(i, i+1)
    - slice -> se usa directo
    - str (alias) -> lo resuelve el llamador antes de traer el tensor
    """
    if isinstance(key, int):
        return t[:, key:key+1]
    elif isinstance(key, slice):
        return t[:, key]
    else:
        raise TypeError("Se esperaba int o slice tras resolver alias de estado.")


def _resolve_state_slice_from_alias(blk: Block, key: AliasKey) -> AliasKey:
    """
    Convierte alias de estado (str) en int/slice usando blk.state_alias.
    Si ya es int/slice, se devuelve tal cual.
    """
    if isinstance(key, (int, slice)):
        return key
    if not isinstance(key, str):
        raise TypeError("Clave de estado debe ser alias (str), int o slice.")
    if key not in blk.state_alias:
        raise KeyError(f"{blk.__class__.__name__}: alias de estado '{key}' no existe.")
    return blk.state_alias[key]


def _parse_signal(sig: str) -> Tuple[str, str]:
    """
    'bloque.puerto' -> ('bloque', 'puerto')
    """
    if "." not in sig:
        raise ValueError(f"Signal '{sig}' debe tener formato 'block.port'.")
    b, p = sig.split(".", 1)
    return b, p


class BaseRecorder:
    """
    Interfaz base de grabadores para Simulator.simulate(...).

    Selección:
    ----------
    - signals: lista de 'block.port' para grabar salidas de outs (puede usar alias de salida).
    - states:  dict {block: [alias|int|slice, ...]} para grabar columnas del estado.

    Notas:
    ------
    - Se asume que todos los tensores tienen batch B y que el batch no cambia.
    - La dimensión temporal se apila como primer eje (T, B, n).
    """

    def __init__(
        self,
        model: Model,
        *,
        signals: Optional[List[str]] = None,
        states: Optional[Mapping[str, List[AliasKey]]] = None,
        store_time: bool = True,
        detach_to_cpu: bool = True,
    ):
        if not model._built:
            raise RuntimeError("Recorder: el Model debe estar 'build()' antes de crear el recorder.")
        self.model = model
        self.signals = [(_parse_signal(s)[0], _parse_signal(s)[1]) for s in (signals or [])]
        self.states_sel: Dict[str, List[AliasKey]] = {bk: list(lst) for bk, lst in (states or {}).items()}
        self.store_time = bool(store_time)
        self.detach_to_cpu = bool(detach_to_cpu)

        # caches para acelerar resolución de alias de salida
        self._out_canon_cache: Dict[Tuple[str, str], str] = {}

        # metadatos
        self._B: Optional[int] = None

    # --------- utilidades de resolución ----------
    def _canon_out(self, block_name: str, port: str) -> str:
        key = (block_name, port)
        if key in self._out_canon_cache:
            return self._out_canon_cache[key]
        blk = self.model.blocks[block_name]
        canon = blk._canon_out(port)
        self._out_canon_cache[key] = canon
        return canon

    def _get_out_tensor(self, outs: Mapping[str, Mapping[str, torch.Tensor]], block_name: str, port: str) -> torch.Tensor:
        canon = self._canon_out(block_name, port)
        return outs[block_name][canon]

    def _get_state_tensor(self, states: Mapping[str, torch.Tensor], block_name: str, key: AliasKey) -> torch.Tensor:
        blk = self.model.blocks[block_name]
        resolved = _resolve_state_slice_from_alias(blk, key)
        return _ensure_2d_by_index(states[block_name], resolved)

    def _prep_tensor(self, t: torch.Tensor) -> torch.Tensor:
        if self.detach_to_cpu:
            return t.detach().to("cpu")
        return t

    # --------- API que usa Simulator.simulate ---------
    # Por diseño, soportamos la firma completa: on_step(t, outs, states)
    # Simulator también soporta on_step(outs, states); aquí implementamos la completa.
    def on_step(self, t: float, outs: Mapping[str, Mapping[str, torch.Tensor]], states: Mapping[str, torch.Tensor]):
        raise NotImplementedError

    # (opcional) para flush/finalizar recursos (archivos, etc.)
    def close(self):
        pass


class MemoryRecorder(BaseRecorder):
    """
    Almacena en memoria las señales seleccionadas.

    - results_signals: dict nombre -> List[Tensor(B,n)]
      donde nombre = 'block.port' (usa alias si lo diste en 'signals').
    - results_states: dict nombre -> List[Tensor(B,k)]
      donde nombre = 'block.state:<alias|i|slice>'
    - times: List[float]
    """

    def __init__(
        self,
        model: Model,
        *,
        signals: Optional[List[str]] = None,
        states: Optional[Mapping[str, List[AliasKey]]] = None,
        store_time: bool = True,
        detach_to_cpu: bool = True,
    ):
        super().__init__(model, signals=signals, states=states, store_time=store_time, detach_to_cpu=detach_to_cpu)
        self.results_signals: Dict[str, List[torch.Tensor]] = {}
        self.results_states: Dict[str, List[torch.Tensor]] = {}
        self.times: List[float] = []

    def on_step(self, t: float, outs, states):
        # Inicializar B si es la primera vez
        if self._B is None and states:
            any_st = next(iter(states.values()))
            self._B = int(any_st.shape[0])

        # señales (outs)
        for (bname, port) in self.signals:
            ten = self._get_out_tensor(outs, bname, port)
            ten = self._prep_tensor(ten)
            key = f"{bname}.{port}"
            self.results_signals.setdefault(key, []).append(ten)

        # estados
        for bname, keys in self.states_sel.items():
            for key in keys:
                ten = self._get_state_tensor(states, bname, key)
                ten = self._prep_tensor(ten)
                label = f"{bname}.state:{key}"
                self.results_states.setdefault(label, []).append(ten)

        if self.store_time:
            self.times.append(float(t))

    # helpers para exportar en batch (T,B,n)
    def stacked(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, lst in self.results_signals.items():
            arr = torch.stack(lst, dim=0).numpy()  # (T,B,n)
            out[k] = arr
        for k, lst in self.results_states.items():
            arr = torch.stack(lst, dim=0).numpy()  # (T,B,k)
            out[k] = arr
        if self.store_time:
            out["t"] = np.asarray(self.times, dtype=np.float64)
        return out


class NPZChunkRecorder(BaseRecorder):
    """
    Grabador 'streaming' por chunks a disco en formato .npz.

    Limitación: .npz no permite append directo; se guardan ficheros por trozos:
        path_pattern.format(part=1) -> bloque 0..chunk-1
        path_pattern.format(part=2) -> bloque chunk..2*chunk-1
        ...
    Al final puedes combinar si lo deseas.

    Estructura de cada .npz:
        - t                (N,)            si store_time=True
        - outs/<blk>.<p>   (N,B,n)
        - state/<blk>:<key>(N,B,k)

    Ejemplo de path_pattern: "runs/msd_trace_part{part:04d}.npz"
    """

    def __init__(
        self,
        model: Model,
        path_pattern: str,
        *,
        chunk_size: int = 500,
        signals: Optional[List[str]] = None,
        states: Optional[Mapping[str, List[AliasKey]]] = None,
        store_time: bool = True,
        detach_to_cpu: bool = True,
        compress: bool = True,
    ):
        super().__init__(model, signals=signals, states=states, store_time=store_time, detach_to_cpu=detach_to_cpu)
        if "{part" not in path_pattern or "}" not in path_pattern:
            raise ValueError("path_pattern debe contener un campo '{part:...}' para numerar los chunks.")
        self.path_pattern = path_pattern
        self.chunk_size = int(chunk_size)
        self.compress = bool(compress)

        # buffers del chunk actual
        self._buf_signals: Dict[str, List[torch.Tensor]] = {}
        self._buf_states: Dict[str, List[torch.Tensor]] = {}
        self._buf_t: List[float] = []
        self._part_idx: int = 1
        self._step_in_chunk: int = 0

        # preparar carpetas
        folder = os.path.dirname(path_pattern.format(part=0))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def _flush_chunk(self):
        if self._step_in_chunk == 0:
            return  # nada que escribir

        arrays: Dict[str, np.ndarray] = {}

        # tiempos
        if self.store_time:
            arrays["t"] = np.asarray(self._buf_t, dtype=np.float64)

        # outs
        for key, lst in self._buf_signals.items():
            arr = torch.stack(lst, dim=0).numpy()  # (N,B,n)
            arrays[f"outs/{key}"] = arr

        # states
        for key, lst in self._buf_states.items():
            arr = torch.stack(lst, dim=0).numpy()  # (N,B,k)
            arrays[f"state/{key}"] = arr

        path = self.path_pattern.format(part=self._part_idx)
        if self.compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)

        # reset buffers del chunk
        self._buf_signals.clear()
        self._buf_states.clear()
        self._buf_t.clear()
        self._step_in_chunk = 0
        self._part_idx += 1

    def on_step(self, t: float, outs, states):
        # Inicializar B si es la primera vez
        if self._B is None and states:
            any_st = next(iter(states.values()))
            self._B = int(any_st.shape[0])

        # señales
        for (bname, port) in self.signals:
            ten = self._get_out_tensor(outs, bname, port)
            ten = self._prep_tensor(ten)
            key = f"{bname}.{port}"
            self._buf_signals.setdefault(key, []).append(ten)

        # estados
        for bname, keys in self.states_sel.items():
            for key in keys:
                ten = self._get_state_tensor(states, bname, key)
                ten = self._prep_tensor(ten)
                label = f"{bname}.state:{key}"
                self._buf_states.setdefault(label, []).append(ten)

        if self.store_time:
            self._buf_t.append(float(t))

        self._step_in_chunk += 1
        if self._step_in_chunk >= self.chunk_size:
            self._flush_chunk()

    def close(self):
        # Escribir última fracción
        self._flush_chunk()
