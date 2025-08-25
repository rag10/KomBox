# externals.py
from typing import Dict, Callable, Optional, Sequence, Tuple, Any, List
import math
import torch

ExtDict = Dict[str, Dict[str, torch.Tensor]]
ExtFn   = Callable[[float, int], ExtDict]

def _as_batch_matrix(value: Any, B: int, width: int,
                     device: Optional[torch.device], dtype: Optional[torch.dtype]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value.to(device=device, dtype=dtype)
        # Se asume forma (B, width). Haz reshape si es necesario.
        return t
    # Escalares o listas -> tensor (B, width)
    t = torch.tensor(value, device=device, dtype=dtype)
    if t.dim() == 0:
        t = t.expand(B, width).clone()
    elif t.dim() == 1 and t.numel() == width:
        t = t.unsqueeze(0).expand(B, width).clone()
    elif t.shape == (B, width):
        pass
    else:
        raise ValueError(f"Valor con forma incompatible para (B={B}, width={width}): {t.shape}")
    return t

def _pack(ext_name: str, port_name: str, tensor: torch.Tensor) -> ExtDict:
    """Empaqueta {ext_name: {port_name: tensor}}. 'port_name' es obligatorio."""
    if port_name is None:
        raise ValueError(f"Se requiere 'port_name' explícito para external '{ext_name}'.")
    return {ext_name: {port_name: tensor}}

def make_external_constant(ext_name: str, *,
                           port: str,
                           value: Any = 0.0,
                           batch_size: int = 1,
                           width: int = 1,
                           device: Optional[torch.device] = None,
                           dtype: Optional[torch.dtype] = None) -> ExtFn:
    """Señal constante (requiere 'port')."""
    tpl = _as_batch_matrix(value, batch_size, width, device, dtype)
    def fn(t: float, k: int) -> ExtDict:
        return _pack(ext_name, port, tpl)
    return fn

def make_external_step(ext_name: str, *,
                       port: str,
                       t0: float = 0.0,
                       y_before: Any = 0.0,
                       y_after: Any  = 1.0,
                       batch_size: int = 1,
                       width: int = 1,
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None) -> ExtFn:
    """Escalón: y = y_before (t<t0), y_after (t>=t0). Requiere 'port'."""
    yb = _as_batch_matrix(y_before, batch_size, width, device, dtype)
    ya = _as_batch_matrix(y_after,  batch_size, width, device, dtype)
    def fn(t: float, k: int) -> ExtDict:
        return _pack(ext_name, port, ya if t >= t0 else yb)
    return fn

def make_external_ramp(ext_name: str, *,
                       port: str,
                       t0: float = 0.0,
                       y0: Any = 0.0,
                       slope: Any = 1.0,
                       clamp: Optional[Tuple[float, float]] = None,
                       batch_size: int = 1,
                       width: int = 1,
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None) -> ExtFn:
    """Rampa: y = y0 + slope*(t-t0), con opcional clamp. Requiere 'port'."""
    y0m    = _as_batch_matrix(y0,    batch_size, width, device, dtype)
    slopem = _as_batch_matrix(slope, batch_size, width, device, dtype)
    def fn(t: float, k: int) -> ExtDict:
        dt = max(0.0, t - t0)
        y  = y0m + slopem * dt
        if clamp is not None:
            y = y.clamp(clamp[0], clamp[1])
        return _pack(ext_name, port, y)
    return fn

def make_external_sine(ext_name: str, *,
                       port: str,
                       amplitude: Any = 1.0,
                       freq_hz: float = 1.0,
                       phase_rad: float = 0.0,
                       offset: Any = 0.0,
                       t_start: float = 0.0,
                       batch_size: int = 1,
                       width: int = 1,
                       device: Optional[torch.device] = None,
                       dtype: Optional[torch.dtype] = None) -> ExtFn:
    """Senoidal con offset y fase, arranca en t_start. Requiere 'port'."""
    off = _as_batch_matrix(offset,    batch_size, width, device, dtype)
    amp = _as_batch_matrix(amplitude, batch_size, width, device, dtype)
    two_pi_f = 2.0 * math.pi * float(freq_hz)
    def fn(t: float, k: int) -> ExtDict:
        if t < t_start:
            return _pack(ext_name, port, off)
        arg = two_pi_f * (t - t_start) + float(phase_rad)
        s = torch.sin(torch.tensor(arg, device=device, dtype=dtype))
        return _pack(ext_name, port, off + amp * s)
    return fn

def make_external_square(ext_name: str, *,
                         port: str,
                         amplitude: Any = 1.0,
                         freq_hz: float = 1.0,
                         duty: float = 0.5,
                         offset: Any = 0.0,
                         t_start: float = 0.0,
                         batch_size: int = 1,
                         width: int = 1,
                         device: Optional[torch.device] = None,
                         dtype: Optional[torch.dtype] = None) -> ExtFn:
    """Cuadrada con duty. Arranca en LOW hasta t_start. Requiere 'port'."""
    off = _as_batch_matrix(offset,    batch_size, width, device, dtype)
    amp = _as_batch_matrix(amplitude, batch_size, width, device, dtype)
    T   = 1.0 / float(freq_hz)
    def fn(t: float, k: int) -> ExtDict:
        if t < t_start:
            return _pack(ext_name, port, off - amp)  # arranca en "low"
        tau  = (t - t_start) % T
        high = (tau / T) < duty
        return _pack(ext_name, port, off + (amp if high else -amp))
    return fn

def combine_externals(*fns: ExtFn) -> ExtFn:
    """Combina varias fns de externals en una sola (merge por claves)."""
    def fn(t: float, k: int) -> ExtDict:
        out: ExtDict = {}
        for f in fns:
            part = f(t, k)
            for name, d in part.items():
                out.setdefault(name, {}).update(d)
        return out
    return fn
