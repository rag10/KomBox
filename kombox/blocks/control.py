# kombox/blocks/control.py
from __future__ import annotations
import torch
from kombox.core.block import DiscreteBlock, ContinuousBlock

class StateSpace(ContinuousBlock):
    """
    x' = A x + B u
    y  = C x + D u
    A:(n,n), B:(n,mu), C:(ny,n), D:(ny,mu)
    inputs: vector 'u' (mu)
    outputs: vector 'y' (ny)
    state: x (n)
    """
    def __init__(self, A, B, C, D):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.get_default_dtype())
        B = torch.as_tensor(B, dtype=torch.get_default_dtype())
        C = torch.as_tensor(C, dtype=torch.get_default_dtype())
        D = torch.as_tensor(D, dtype=torch.get_default_dtype())
        n, mu = A.shape[0], B.shape[1]
        ny    = C.shape[0]
        self.declare_io(inputs={"u":mu}, outputs={"y":ny}, state_size=n)
        # guardamos matrices como parámetros no entrenables
        self.declare_param("A", A); self.declare_param("B", B)
        self.declare_param("C", C); self.declare_param("D", D)

    def ode(self, state, inputs, t):
        x = state
        u = inputs["u"]
        A = self.get_param("A"); B = self.get_param("B")
        C = self.get_param("C"); D = self.get_param("D")
        # broadcast por batch: (B,n) @ (n,n)^T no vale; expandimos con einsum
        dx = torch.einsum("ij,bj->bi", A, x) + torch.einsum("ij,bj->bi", B, u)
        y  = torch.einsum("ij,bj->bi", C, x) + torch.einsum("ij,bj->bi", D, u)
        return dx, {"y": y}

class PID(DiscreteBlock):
    """
    PID discreto en forma posicional:
        u = Kp*e + I + Kd*(e - e_prev)/dt
        I_{k+1} = I_k + Ki*dt*e  (con anti-windup por "conditional integration")
    Estados: [I, e_prev]  -> state_size=2
    Entradas: e (1)
    Salidas:  u (1) limitada a [u_min, u_max]
    """
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, u_min: float = -float("inf"), u_max: float = float("inf")):
        super().__init__()
        self.declare_io(inputs={"e": 1}, outputs={"u": 1}, state_size=2)
        self.declare_param("Kp", Kp); self.declare_param("Ki", Ki); self.declare_param("Kd", Kd)
        self.declare_param("u_min", float(u_min)); self.declare_param("u_max", float(u_max))
        self.state_alias = {"I": 0, "e_prev": 1}

    def update(self, state, inputs, dt, t):
        e = inputs["e"]  # (B,1)
        I = state[:, 0:1]
        e_prev = state[:, 1:2]

        Kp = self.get_param("Kp"); Ki = self.get_param("Ki"); Kd = self.get_param("Kd")
        u_min = self.get_param("u_min"); u_max = self.get_param("u_max")

        # Derivada discreta simple (sin filtro)
        de = (e - e_prev) / max(float(dt), 1e-12)

        # Tentativo antes de anti-windup (para decidir si integramos)
        u_tent = Kp * e + I + Kd * de
        u_sat = torch.clamp(u_tent, u_min, u_max)

        # Anti-windup: integra solo si no está saturado o si el error empuja hacia adentro
        integrating = (u_tent == u_sat) | ((u_tent > u_sat) & (e < 0.0)) | ((u_tent < u_sat) & (e > 0.0))
        I_new = I + (Ki * float(dt) * e) * integrating

        # Salida final
        u = torch.clamp(Kp * e + I_new + Kd * de, u_min, u_max)

        new_state = torch.cat([I_new, e], dim=1)
        return new_state, {"u": u}
