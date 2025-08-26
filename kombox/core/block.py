# kombox/core/block.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple, Any, Optional, Union
import warnings
import torch
import torch.nn as nn


@dataclass(frozen=True)
class PortSpec:
    """
    Especificación mínima de un puerto:
      - n: ancho del puerto (canales por muestra). Las señales son (B, n).
    """
    n: int


class Block(nn.Module):
    """
    Clase base de todos los bloques KomBox.

    Conceptos
    ---------
    Puertos (IO)
      - declare_io(inputs=..., outputs=..., state_size=..., repeated_inputs=..., repeated_outputs=...)
      - Alias: alias_inputs({alias:canon}), alias_outputs({alias:canon})
        * Un único alias por puerto canónico. Sobrescribe con warning si ya existía.
      - Resolución de nombres de puerto escritos por el usuario → canónicos:
        * _canon_in(name), _canon_out(name)

    Estado
      - Tamaño total S vía declare_io(state_size=S).
      - init_state(B) → (B, S) (por defecto ceros; si S=0, retorna (B,0)).
      - state_alias: dict opcional {nombre: slice/int} para inicialización legible.

    Parámetros (API simplificada)
      - declare_param(name, value, trainable=False)
          * value: escalar, vector (D,), o matriz (B,D) → se normaliza SIEMPRE a 2D.
          * por defecto no entrenable (buffer). Si trainable=True → nn.Parameter.
      - set_param(name, value)
          * acepta escalar/vector/2D; valida columnas D contra la declaración inicial.
      - expand_param(name, batch_size)
          * (1,D) → (B,D). Si ya era (B,D), no hace nada.
      - make_param_trainable(name, True/False)
          * convierte entre buffer y nn.Parameter conservando el valor actual.
      - set_param_trainable(name, ...) → alias del anterior.
      - get_param(name) → Tensor 2D (B,D)

    Validación de batch
      - No se fuerza en set_param/expand_param; se comprueba en Model.initialize(B),
        donde cada parámetro debe tener B ∈ {1, batch_size}.

    API de simulación (a implementar en subclases)
      - outputs(state, inputs, t) -> dict[str, Tensor] (por defecto usa step(..., dt=0))
      - deriv(state, inputs, t)   -> (dx, outs) (bloques continuos)
      - step(state, inputs, dt)   -> (new_state, outs) (bloques discretos / Euler simple)
    """

    # ===================== Init =====================
    def __init__(self):
        super().__init__()
        # Puertos
        self.in_specs: Dict[str, PortSpec] = {}
        self.out_specs: Dict[str, PortSpec] = {}

        # Aliases (un alias por canónico; nombres amigables -> canónicos)
        self._in_alias: Dict[str, str] = {}
        self._out_alias: Dict[str, str] = {}

        # Estado
        self._state_size: int = 0
        # Nombres de subestados -> slices/índices (para inicialización legible)
        self.state_alias: Dict[str, Union[slice, int]] = {}

        # Parámetros
        # name -> atributo donde vive el tensor (buffer o Parameter)
        self._param_attr: Dict[str, str] = {}
        # name -> número de columnas D (ancho "por muestra")
        self._param_width: Dict[str, int] = {}
        # name -> flag entrenable
        self._param_trainable: Dict[str, bool] = {}

    # ===================== Puertos =====================
    def declare_io(
        self,
        inputs: Optional[Mapping[str, int]] = None,
        outputs: Optional[Mapping[str, int]] = None,
        state_size: Optional[int] = None,
        *,
        repeated_inputs: Optional[Mapping[str, Tuple[int, int]]] = None,
        repeated_outputs: Optional[Mapping[str, Tuple[int, int]]] = None,
    ):
        """
        Declaración compacta de puertos y estado.

        Args:
            inputs: dict {nombre: ancho} para entradas (señales (B, ancho)).
            outputs: dict {nombre: ancho} para salidas (señales (B, ancho)).
            state_size: dimensión total del estado S (opcional; por defecto 0).
            repeated_inputs: dict {prefijo: (count, size)} -> genera entradas prefijo1..prefijoN.
            repeated_outputs: dict {prefijo: (count, size)} -> genera salidas prefijo1..prefijoN.
        """
        if inputs:
            for n, s in inputs.items():
                self.in_specs[n] = PortSpec(int(s))
        if outputs:
            for n, s in outputs.items():
                self.out_specs[n] = PortSpec(int(s))
        if repeated_inputs:
            for pref, (count, size) in repeated_inputs.items():
                for i in range(1, count + 1):
                    self.in_specs[f"{pref}{i}"] = PortSpec(int(size))
        if repeated_outputs:
            for pref, (count, size) in repeated_outputs.items():
                for i in range(1, count + 1):
                    self.out_specs[f"{pref}{i}"] = PortSpec(int(size))
        if state_size is not None:
            self._state_size = int(state_size)
        return self

    # ----- Aliases con validación y sobrescritura controlada -----
    def _specs_for(self, direction: str) -> Dict[str, PortSpec]:
        if direction == "in":  return self.in_specs
        if direction == "out": return self.out_specs
        raise ValueError("direction debe ser 'in' o 'out'")

    def _aliases_for(self, direction: str) -> Dict[str, str]:
        if direction == "in":  return self._in_alias
        if direction == "out": return self._out_alias
        raise ValueError("direction debe ser 'in' o 'out'")

    def _validate_and_set_aliases(self, direction: str, mapping: Mapping[str, str]):
        """
        Reglas:
          - El canónico debe existir en in_specs/out_specs.
          - alias != canónico.
          - alias no puede colisionar con un nombre canónico existente.
          - Máximo 1 alias por canónico: si ya existía, se SOBREESCRIBE y se avisa con warnings.warn.
          - Si el alias existía para otro canónico, se avisa y se sobreescribe.
        """
        specs = self._specs_for(direction)
        amap  = self._aliases_for(direction)

        # Inverso actual canónico -> alias (si lo hay)
        canon_to_alias = {canon: alias for alias, canon in amap.items()}

        for alias, canon in mapping.items():
            if alias == canon:
                raise ValueError(f"[{direction}] alias '{alias}' no puede ser igual al canónico.")
            if canon not in specs:
                raise KeyError(f"[{direction}] puerto canónico '{canon}' no existe.")
            if alias in specs:
                raise ValueError(f"[{direction}] alias '{alias}' colisiona con nombre canónico existente.")

            prev_alias_for_canon = canon_to_alias.get(canon)
            if prev_alias_for_canon is not None and prev_alias_for_canon != alias:
                warnings.warn(
                    f"[{direction}] sobrescribiendo alias '{prev_alias_for_canon}' -> '{canon}' por nuevo alias '{alias}'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                amap.pop(prev_alias_for_canon, None)

            prev_canon_for_alias = amap.get(alias)
            if prev_canon_for_alias is not None and prev_canon_for_alias != canon:
                warnings.warn(
                    f"[{direction}] alias '{alias}' cambia destino: '{prev_canon_for_alias}' -> '{canon}'.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            amap[alias] = canon

        return self

    def alias_inputs(self, mapping: Mapping[str, str]):
        """Registra alias de entrada {alias -> canónico} con validación y sobrescritura controlada."""
        return self._validate_and_set_aliases("in", mapping)

    def alias_outputs(self, mapping: Mapping[str, str]):
        """Registra alias de salida {alias -> canónico} con validación y sobrescritura controlada."""
        return self._validate_and_set_aliases("out", mapping)

    def _canon_in(self, name: str) -> str:
        """Traduce nombre de entrada potencialmente 'amigable' al canónico."""
        return self._in_alias.get(name, name)

    def _canon_out(self, name: str) -> str:
        """Traduce nombre de salida potencialmente 'amigable' al canónico."""
        return self._out_alias.get(name, name)

    # ===================== Estado =====================
    def state_size(self) -> int:
        """Devuelve S, dimensión total del estado del bloque."""
        return self._state_size

    def init_state(self, batch_size: int, *, device=None, dtype=None) -> torch.Tensor:
        """
        Crea el tensor de estado (B, S).
        - S=0: tensor vacío (B,0) para uniformidad.
        - S>0: ceros por defecto. Sobrescribe si tu bloque necesita otra cosa.
        """
        S = self._state_size
        if S == 0:
            return torch.empty((batch_size, 0), device=device, dtype=dtype)
        return torch.zeros((batch_size, S), device=device, dtype=dtype)
    
    def _complete_inputs_with_zeros(
        self,
        state: torch.Tensor,
        inputs: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Devuelve un dict con TODAS las entradas declaradas:
        - copia las que existan en `inputs`
        - rellena las que falten con ceros (B, n) basados en in_specs
        Se usa en Fase A para evitar KeyError en bloques que leen entradas para dx.
        """
        B = state.shape[0]
        device = state.device
        dtype = state.dtype
        full: Dict[str, torch.Tensor] = dict(inputs)  # copia superficial
        for name, spec in self.in_specs.items():
            if name not in full:
                full[name] = torch.zeros((B, spec.n), device=device, dtype=dtype)
        return full


    # ===================== Parámetros (API simple) =====================
    def _normalize_param_shape(
        self,
        value: torch.Tensor | float | list,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Normaliza a 2D (B, D):
          - escalar -> (1,1)
          - vector  -> (1,D)
          - 2D      -> (B,D)
        """
        t = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
        if device is not None:
            t = t.to(device)
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t.unsqueeze(0)
        elif t.ndim == 2:
            pass
        else:
            raise ValueError(f"Parámetro debe ser escalar, vector o 2D (B,D). Recibido shape={tuple(t.shape)}")
        return t

    def _set_param_tensor(self, name: str, t: torch.Tensor):
        """
        Escribe el tensor t en el atributo correspondiente conservando el tipo:
          - si era Parameter -> actualiza .data
          - si era buffer    -> reemplaza buffer manteniendo device/dtype
          - si no existía    -> registra como buffer por defecto
        """
        attr = self._param_attr[name]
        holder = getattr(self, attr, None)
        if isinstance(holder, nn.Parameter):
            holder.data = t.to(device=holder.device, dtype=holder.dtype)
        elif isinstance(holder, torch.Tensor):
            device, dtype = holder.device, holder.dtype
            setattr(self, attr, t.to(device=device, dtype=dtype))
        else:
            self.register_buffer(attr, t)

    def declare_param(self, name: str, value: torch.Tensor | float | list, *, trainable: bool = False):
        """
        Declara un parámetro:
          - no entrenable por defecto (buffer).
          - se guarda como 2D (B,D); típicamente (1,D).
          - si trainable=True, se crea nn.Parameter.
        """
        if name in self._param_attr:
            raise ValueError(f"Parámetro '{name}' ya declarado.")
        t = self._normalize_param_shape(value)
        B, D = t.shape
        attr = f"param_{name}"
        if trainable:
            p = nn.Parameter(t.clone().detach(), requires_grad=True)
            setattr(self, attr, p)
        else:
            self.register_buffer(attr, t.clone().detach())
        self._param_attr[name] = attr
        self._param_width[name] = D
        self._param_trainable[name] = bool(trainable)
        return self

    def set_param(self, name: str, value: torch.Tensor | float | list):
        """
        Asigna un nuevo valor (escalares, vectores o batch).
        No valida contra batch_size global (eso se hace en Model.initialize).
        Comprueba que el número de columnas D coincide con la declaración inicial.
        """
        if name not in self._param_attr:
            raise KeyError(f"Parámetro '{name}' no existe. Decláralo antes con declare_param().")
        t = self._normalize_param_shape(value)
        _, D = t.shape
        if D != self._param_width[name]:
            raise ValueError(f"{self.__class__.__name__}.{name}: D={D} no coincide con D declarado={self._param_width[name]}")
        self._set_param_tensor(name, t)
        return self

    def expand_param(self, name: str, batch_size: int):
        """
        Expande (1,D) -> (B,D). Si ya es (B,D), no hace nada.
        Otros casos incompatibles se detectarán en Model.initialize(B).
        """
        if name not in self._param_attr:
            raise KeyError(f"Parámetro '{name}' no existe.")
        attr = self._param_attr[name]
        t = getattr(self, attr)
        if t.ndim != 2:
            raise RuntimeError(f"{self.__class__.__name__}.{name}: se esperaba 2D (B,D), got {tuple(t.shape)}")
        B, D = t.shape
        if B == 1:
            t_exp = t.expand(int(batch_size), D).clone()
            self._set_param_tensor(name, t_exp)
        return self

    def make_param_trainable(self, name: str, trainable: bool = True):
        """
        Convierte el parámetro a entrenable (nn.Parameter) o no entrenable (buffer).
        Conserva el valor actual.
        """
        if name not in self._param_attr:
            raise KeyError(f"Parámetro '{name}' no existe.")
        attr = self._param_attr[name]
        t = getattr(self, attr).detach()
        if hasattr(self, attr):
            delattr(self, attr)
        if trainable:
            setattr(self, attr, nn.Parameter(t.clone(), requires_grad=True))
        else:
            self.register_buffer(attr, t.clone())
        self._param_trainable[name] = bool(trainable)
        return self

    # Alias por compatibilidad con ejemplos/utilidades
    def set_param_trainable(self, name: str, trainable: bool = True):
        return self.make_param_trainable(name, trainable=trainable)

    def get_param(self, name: str) -> torch.Tensor:
        """Devuelve el tensor 2D (B,D) tal y como está almacenado (B puede ser 1 o el batch)."""
        if name not in self._param_attr:
            raise KeyError(f"Parámetro '{name}' no existe.")
        return getattr(self, self._param_attr[name])

    # ===================== Validación de señales (usadas por Simulator) =====================
    def _check_tensor_matrix(self, t: torch.Tensor, *, rows: int, cols: int, kind: str, name: str):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{self.__class__.__name__}.{kind} '{name}': debe ser torch.Tensor.")
        if t.ndim != 2:
            raise ValueError(f"{self.__class__.__name__}.{kind} '{name}': esperado 2D (B,{cols}), recibido {tuple(t.shape)}.")
        if t.shape[0] != rows:
            raise ValueError(f"{self.__class__.__name__}.{kind} '{name}': batch esperado={rows}, recibido={t.shape[0]}.")
        if t.shape[1] != cols:
            raise ValueError(f"{self.__class__.__name__}.{kind} '{name}': ancho esperado={cols}, recibido={t.shape[1]}.")

    def _validate_inputs(self, inputs: Mapping[str, torch.Tensor], *, batch_size: int):
        # Comprueba que están todas las entradas declaradas y sin extras
        declared = set(self.in_specs.keys())
        got = set(inputs.keys())
        missing = declared - got
        extra = got - declared
        if missing:
            raise KeyError(f"{self.__class__.__name__}: faltan entradas {sorted(missing)}.")
        if extra:
            raise KeyError(f"{self.__class__.__name__}: entradas no declaradas {sorted(extra)}.")
        # Chequea shapes
        for name, spec in self.in_specs.items():
            self._check_tensor_matrix(inputs[name], rows=batch_size, cols=spec.n, kind="input", name=name)

    def _validate_outputs(self, outputs: Mapping[str, torch.Tensor], *, batch_size: int):
        declared = set(self.out_specs.keys())
        got = set(outputs.keys())
        missing = declared - got
        extra = got - declared
        if missing:
            raise KeyError(f"{self.__class__.__name__}: faltan salidas {sorted(missing)}.")
        if extra:
            raise KeyError(f"{self.__class__.__name__}: salidas no declaradas {sorted(extra)}.")
        for name, spec in self.out_specs.items():
            self._check_tensor_matrix(outputs[name], rows=batch_size, cols=spec.n, kind="output", name=name)

    # ---- DAE/Algebraic (opcional) ----
    def algebraic_residual(
        self,
        t: float,
        state: Optional[torch.Tensor],
        inputs: Mapping[str, torch.Tensor] | Dict[str, torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor] | torch.Tensor:
        """
        Residuales algebraicos locales g(...)=0 del bloque (si aplica).
        Por defecto no hay ecuaciones algebraicas y devuelve {}.
        Mantener todo en torch (autograd-safe).
        """
        return {}

# --- API simplificada por tipo de bloque --------------------------------------
class ContinuousBlock(Block):
    """Bloque continuo (ODE): define solo `ode(state, inputs, t) -> (dx, outs)`."""
    def ode(self, state, inputs, t):
        raise NotImplementedError

    # Core hooks (no los usa el usuario):
    def _expose_outputs(self, state, inputs, t):
        # Completa entradas faltantes con ceros SOLO para fase A
        safe_inputs = self._complete_inputs_with_zeros(state, inputs)
        _, outs = self.ode(state, safe_inputs, t)
        return outs

    def _advance(self, state, inputs, dt, t, solver):
        # El solver decide cómo integrar (Euler, RK4, torchdiffeq…)
        return solver.step_continuous(self, state, inputs, float(dt), float(t))


class DiscreteBlock(Block):
    """Bloque discreto: define solo `update(state, inputs, dt, t) -> (new_state, outs)`."""
    def update(self, state, inputs, dt, t):
        raise NotImplementedError

    # Core hooks:
    def _expose_outputs(self, state, inputs, t):
        safe_inputs = self._complete_inputs_with_zeros(state, inputs)
        _, outs = self.update(state, safe_inputs, 0.0, t)
        return outs

    def _advance(self, state, inputs, dt, t, solver):
        return self.update(state, inputs, float(dt), float(t))

# ===================== Helpers sencillos =====================
class _FunctionalContinuous(ContinuousBlock):
    def __init__(self, fn, *, inputs, outputs, state_size):
        super().__init__()
        self._fn = fn
        self.declare_io(inputs=inputs, outputs=outputs, state_size=state_size)
    def ode(self, state, inputs, t):
        return self._fn(state, inputs, t)

class _FunctionalDiscrete(DiscreteBlock):
    def __init__(self, fn, *, inputs, outputs, state_size=0):
        super().__init__()
        self._fn = fn
        self.declare_io(inputs=inputs, outputs=outputs, state_size=state_size)
    def update(self, state, inputs, dt, t):
        return self._fn(state, inputs, dt, t)

def from_ode(fn, *, inputs, outputs, state_size):
    """
    Crea un bloque continuo desde una función:
        fn(state, inputs, t) -> (dx, outs)
    """
    return _FunctionalContinuous(fn, inputs=inputs, outputs=outputs, state_size=state_size)

def from_update(fn, *, inputs, outputs, state_size=0):
    """
    Crea un bloque discreto desde una función:
        fn(state, inputs, dt, t) -> (new_state, outs)
    """
    return _FunctionalDiscrete(fn, inputs=inputs, outputs=outputs, state_size=state_size)
