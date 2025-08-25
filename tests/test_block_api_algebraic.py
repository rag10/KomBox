import torch
from kombox.core.block import Block

def test_block_algebraic_residual_default_empty():
    b = Block()
    res = b.algebraic_residual(0.0, None, {}, {})
    assert isinstance(res, (dict, torch.Tensor))
    if isinstance(res, dict): assert len(res) == 0
    else: assert res.numel() == 0
