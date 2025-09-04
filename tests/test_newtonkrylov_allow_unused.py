# tests/test_newtonkrylov_allow_unused.py
import torch
import pytest

from kombox.core.algebraic.newton_krylov import NewtonKrylov

@pytest.mark.parametrize("B,Zdim,Rout", [(3, 5, 4), (2, 1, 1)])
def test_vjp_allow_unused_returns_zero(B, Zdim, Rout):
    """
    Verifica que la VJP (J^T v) no explota cuando F(z) NO depende de z.
    En ese caso, autograd devuelve None y esperamos que el método devuelva ceros.
    """
    torch.set_default_dtype(torch.float32)
    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=5)

    def F(z: torch.Tensor) -> torch.Tensor:
        # Residual independiente de z: forma (B, Rout)
        return torch.arange(Rout, dtype=z.dtype, device=z.device).view(1, Rout).expand(z.shape[0], Rout)

    z = torch.randn(B, Zdim, requires_grad=True)
    v = torch.randn(B, Rout)

    JT_v = alg._vjp(F, z, v)  # debe NO lanzar y devolver ceros
    assert JT_v.shape == z.shape
    assert torch.allclose(JT_v, torch.zeros_like(z)), "Cuando F no depende de z, J^T v debe ser cero."


def test_solver_with_constant_zero_residual_no_crash():
    """
    Smoke: si F(z) == 0 para todo z (independiente), JFNK no debe fallar y
    debe devolver algo con la forma correcta (normalmente z0).
    """
    torch.set_default_dtype(torch.float32)
    B, Zdim, Rout = 4, 3, 2
    alg = NewtonKrylov(mode="jfnk", tol=1e-10, max_iter=5)

    def F_zero(z: torch.Tensor) -> torch.Tensor:
        return torch.zeros((z.shape[0], Rout), dtype=z.dtype, device=z.device)

    z0 = torch.randn(B, Zdim, requires_grad=True)
    z_star = alg.solve(F_zero, z0)

    assert z_star.shape == z0.shape
    assert torch.isfinite(z_star).all(), "El solver debe devolver valores finitos."
