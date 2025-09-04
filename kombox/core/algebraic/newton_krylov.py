# kombox/core/algebraic/newton_krylov.py
from __future__ import annotations
from typing import Callable, Optional, Tuple
import torch
from torch import Tensor


def _l2_norm(t: Tensor) -> Tensor:
    # Norma por batch (B,) o escalar si B=1
    if t.ndim == 2:
        return torch.sqrt(torch.sum(t * t, dim=1) + 1e-16)
    return torch.linalg.norm(t)


class NewtonKrylov:
    """
    Solver algebraico F(z)=0 autograd-safe.

    Modos:
      - mode="dense":   construye J explícito por batch (tu comportamiento actual).
      - mode="jfnk":    Jacobian-Free Newton–Krylov (JVP/VJP + CG sobre (J^T J + λI)).

    Uso:
      solver = NewtonKrylov(mode="dense")   # retrocompatible
      solver = NewtonKrylov(mode="jfnk")    # escalable a z grande
      z_star = solver.solve(F, z0)
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 20,
        line_search: bool = True,
        # Selección de modo:
        mode: str = "dense",
        # Parámetros JFNK:
        damping: float = 1e-8,      # λ en J^T J + λ I
        cg_maxit: int = 50,
        cg_tol: float = 1e-8,
        ls_shrink: float = 0.5,
        ls_min: float = 1e-4,
        verbose: bool = False,
    ):
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.line_search = bool(line_search)

        self.mode = str(mode).lower()
        self.damping = float(damping)
        self.cg_maxit = int(cg_maxit)
        self.cg_tol = float(cg_tol)
        self.ls_shrink = float(ls_shrink)
        self.ls_min = float(ls_min)
        self.verbose = bool(verbose)

    # ---------------- API pública ----------------
    def solve(self, F: Callable[[Tensor], Tensor], z0: Tensor) -> Tensor:
        """
        Resuelve F(z)=0 partiendo de z0 (B,Z). Devuelve z* (B,Z).
        Si Z=0 (vector vacío), devuelve z0.
        """
        if z0.shape[1] == 0:
            return z0
        if self.mode == "dense":
            return self._solve_dense(F, z0)
        elif self.mode == "jfnk":
            return self._solve_jfnk(F, z0)
        else:
            raise ValueError(f"NewtonKrylov: modo desconocido '{self.mode}'. Use 'dense' o 'jfnk'.")

    # ---------------- Implementación DENSE (tu comportamiento actual) ----------------
    def _solve_dense(self, F: Callable[[Tensor], Tensor], z0: Tensor) -> Tensor:
        """
        Forma J por batch con autograd.functional.jacobian y resuelve J dz = r; z <- z - dz.
        """
        z = z0.clone().requires_grad_(True)
        for _ in range(self.max_iter):
            r = F(z)  # (B,R)
            if _l2_norm(r) <= self.tol:
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

            # Line search
            step = 1.0
            for _ls in range(6) if self.line_search else range(1):
                z_try = (z - step * dz).detach().requires_grad_(True)
                if _l2_norm(F(z_try)) < _l2_norm(r):
                    z = z_try
                    break
                step *= 0.5
            else:
                z = (z - dz).detach().requires_grad_(True)
        return z

    # ---------------- Implementación JFNK (Jacobian-Free) ----------------
    @torch.enable_grad()
    def _vjp(self, F, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Vector–Jacobian product J(z)^T v computed with autograd.
        - F(z): (B, R)
        - z:    (B, Z)
        - v:    (B, R)
        Returns: (B, Z)

        Robust to the case "F does not depend on z": in that situation ``y = F(z)``
        will not require grad and ``autograd.grad`` would normally error. We detect
        it and return zeros like ``z``. We also pass ``allow_unused=True`` and
        sanitize possible ``None`` gradients.
        """
        with torch.enable_grad():
            z = z.requires_grad_(True)
            y = F(z)
            # If F does not depend on z, y won't require grad → J = 0.
            if not getattr(y, "requires_grad", False):
                return torch.zeros_like(z)
            grad = torch.autograd.grad(
                y,
                z,
                grad_outputs=v,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                return torch.zeros_like(z)
            return grad

    @torch.enable_grad()
    def _jvp(self, F: Callable[[Tensor], Tensor], z: Tensor, v: Tensor) -> Tensor:
        """J v via autograd.functional.jvp; fallback a FD direccional si no disponible."""
        try:
            out, jvp = torch.autograd.functional.jvp(F, z, v, create_graph=True)
            return jvp
        except Exception:
            eps = 1e-6
            return (F(z + eps * v) - F(z)) / eps

    @torch.no_grad()
    def _cg(
        self,
        Aop: Callable[[Tensor], Tensor],
        b: Tensor,
        x0: Optional[Tensor] = None,
        maxit: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Tuple[Tensor, int]:
        """CG sobre A x = b por batch."""
        maxit = self.cg_maxit if maxit is None else maxit
        tol = self.cg_tol if tol is None else tol

        B = b.shape[0]
        x = b.new_zeros(b.shape) if x0 is None else x0.clone()
        r = b - Aop(x)
        p = r.clone()
        rs_old = torch.sum(r * r, dim=1)  # (B,)

        for k in range(maxit):
            Ap = Aop(p)
            alpha = rs_old / (torch.sum(p * Ap, dim=1) + 1e-16)
            x = x + alpha.view(B, 1) * p
            r = r - alpha.view(B, 1) * Ap
            rs_new = torch.sum(r * r, dim=1)
            if torch.sqrt(rs_new.max()) < tol:
                return x, k + 1
            beta = rs_new / (rs_old + 1e-16)
            p = r + beta.view(B, 1) * p
            rs_old = rs_new
        return x, maxit

    def _solve_jfnk(self, F: Callable[[Tensor], Tensor], z0: Tensor) -> Tensor:
        """
        Newton–Krylov con Gauss–Newton + line search:
           (J^T J + λ I) dx = -J^T r
        """
        z = z0.clone().requires_grad_(True)

        for it in range(self.max_iter):
            r = F(z)  # (B,R)
            rn = _l2_norm(r)
            if self.verbose:
                print(f"[JFNK] it={it} ||r||={float(rn.max()):.3e}")
            if float(rn.max()) < self.tol:
                return z

            # Operador A(dx) = J^T J dx + λ dx via JVP/VJP
            def Aop(dx: Tensor) -> Tensor:
                jv = self._jvp(F, z, dx)            # (B,R)
                JT_jv = self._vjp(F, z, jv)         # (B,Z)
                return JT_jv + self.damping * dx

            # RHS = -J^T r
            JT_r = self._vjp(F, z, r)              # (B,Z)
            rhs = -JT_r.detach()

            with torch.no_grad():
                dx, _ = self._cg(Aop, rhs, x0=None)

            # Line search backtracking
            alpha = 1.0
            f0 = _l2_norm(r).detach()
            while alpha > self.ls_min:
                z_try = (z + alpha * dx).detach().requires_grad_(True)
                r_try = F(z_try)
                if _l2_norm(r_try).max() <= f0.max() * (1 - 1e-4 * alpha):
                    z = z_try
                    break
                alpha *= self.ls_shrink
            else:
                z = (z + alpha * dx).detach().requires_grad_(True)

        return z


# Azúcar opcional:
class NewtonDense(NewtonKrylov):
    def __init__(self, **kw):
        super().__init__(mode="dense", **kw)

class NewtonJFNK(NewtonKrylov):
    def __init__(self, **kw):
        super().__init__(mode="jfnk", **kw)
