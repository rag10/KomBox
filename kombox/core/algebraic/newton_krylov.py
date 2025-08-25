# kombox/core/algebraic/newton_krylov.py
from __future__ import annotations
import torch

class NewtonKrylov:
    """
    Solver algebraico F(z)=0 autograd-safe (MVP: Jacobiano denso).
    Sustituible por versión JVP-only + GMRES más adelante.
    """
    def __init__(self, tol: float = 1e-8, max_iter: int = 20, line_search: bool = True):
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.line_search = bool(line_search)

    @staticmethod
    def _norm2(v: torch.Tensor) -> torch.Tensor:
        if v.numel() == 0:
            return torch.zeros((), device=v.device, dtype=v.dtype)
        return torch.linalg.vector_norm(v.reshape(v.shape[0], -1), ord=2, dim=1).max()

    def solve(self, F, z0: torch.Tensor) -> torch.Tensor:
        z = z0.clone().requires_grad_(True)
        for _ in range(self.max_iter):
            r = F(z)  # (B,R)
            if self._norm2(r) <= self.tol:
                return z
            # Jacobiano denso batcheado
            B = r.shape[0]
            rflat = r.reshape(B, -1)
            J_rows = []
            for b in range(B):
                def fb(zb):
                    zz = z.clone()
                    zz[b] = zb.reshape_as(z[b])
                    return F(zz)[b].reshape(-1)
                Jb = torch.autograd.functional.jacobian(fb, z[b], create_graph=True)  # (R,Z)
                J_rows.append(Jb)
            # Resolver por batch
            dz_list = []
            for b in range(B):
                rb = rflat[b]
                Jb = J_rows[b]
                dzb = torch.linalg.solve(Jb, rb)
                dz_list.append(dzb)
            dz = torch.stack(dz_list, dim=0).reshape_as(z)
            # line search simple
            step = 1.0
            for _ls in range(6) if self.line_search else range(1):
                z_try = (z - step*dz).detach().requires_grad_(True)
                if self._norm2(F(z_try)) < self._norm2(r):
                    z = z_try
                    break
                step *= 0.5
            else:
                z = (z - dz).detach().requires_grad_(True)
        return z