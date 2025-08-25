# kombox/tests/test_jfnk_basic.py
import torch
from kombox.core.algebraic.newton_krylov import NewtonKrylov

def test_jfnk_solves_scalar_nonlinear():
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    # F(z) = z^2 - 2 -> raíz ~ sqrt(2)
    def F(z):
        return z**2 - 2.0

    z0 = torch.tensor([[1.0]], device=device)  # (B=1, Z=1)
    solver = NewtonKrylov(max_iter=50, tol=1e-10, damping=1e-8, cg_maxit=64)
    z_star = solver.solve(F, z0)

    assert z_star.shape == z0.shape
    assert torch.allclose(z_star, torch.tensor([[2.0**0.5]]), atol=1e-4, rtol=0)
