from typing import Dict, Any, Union
import torch
import warnings


def validate_tensor_shapes(tensors: Dict[str, torch.Tensor], expected_shapes: Dict[str, tuple]):
    """
    Valida que los tensores tengan las formas esperadas.
    
    Args:
        tensors: Diccionario de tensores a validar
        expected_shapes: Diccionario de formas esperadas {nombre: (dim1, dim2, ...)}
    """
    for name, tensor in tensors.items():
        if name in expected_shapes:
            expected = expected_shapes[name]
            if tensor.shape != expected:
                raise ValueError(f"Tensor {name} tiene forma {tensor.shape}, esperaba {expected}")


def batch_expand(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Expande un tensor para que tenga el batch_size especificado.
    
    Args:
        tensor: Tensor a expandir
        batch_size: Tamaño de batch deseado
    
    Returns:
        Tensor expandido
    """
    if tensor.dim() == 0:
        # Escalar -> (batch_size, 1)
        return tensor.unsqueeze(0).expand(batch_size, 1)
    elif tensor.dim() == 1:
        # Vector -> (batch_size, D)
        return tensor.unsqueeze(0).expand(batch_size, -1)
    elif tensor.dim() == 2:
        if tensor.shape[0] == 1:
            # (1, D) -> (batch_size, D)
            return tensor.expand(batch_size, -1)
        elif tensor.shape[0] == batch_size:
            # Ya tiene el tamaño correcto
            return tensor
        else:
            raise ValueError(f"No se puede expandir tensor de forma {tensor.shape} a batch_size {batch_size}")
    else:
        raise ValueError(f"Tensor debe tener 0, 1 o 2 dimensiones, tiene {tensor.dim()}")


def check_finite(tensor: torch.Tensor, name: str = "tensor"):
    """
    Verifica que un tensor no contenga NaN o Inf.
    
    Args:
        tensor: Tensor a verificar
        name: Nombre del tensor para mensajes de error
    
    Raises:
        ValueError: Si el tensor contiene NaN o Inf
    """
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise ValueError(f"{name} contiene {nan_count} NaN y {inf_count} Inf")


def to_device_dtype(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Mueve tensor a device y dtype especificados si es necesario.
    
    Args:
        tensor: Tensor a convertir
        device: Device destino
        dtype: Dtype destino
    
    Returns:
        Tensor convertido
    """
    if tensor.device != device or tensor.dtype != dtype:
        return tensor.to(device=device, dtype=dtype)
    return tensor


def warn_if_slow_device(device: torch.device):
    """
    Emite warning si se está usando un device lento para simulación.
    """
    if device.type == 'cpu':
        warnings.warn(
            "Simulación ejecutándose en CPU. Para mejor rendimiento, usa GPU si está disponible.",
            UserWarning,
            stacklevel=2
        )