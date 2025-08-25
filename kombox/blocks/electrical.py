# kombox/blocks/electrical.py
from __future__ import annotations
import torch
from kombox.core.block import ContinuousBlock

class DCMotor(ContinuousBlock):
    """
    Modelo clásico:
      L di/dt = V - R i - ke ω
      J dω/dt = kt i - b ω - τ_load
    Estado: [i, ω] ; Entradas: V, tau_load ; Salidas: i, ω
    """
    def __init__(self, R=1.0, L=0.5, ke=0.01, kt=0.01, J=0.01, b=0.1):
        super().__init__()
        self.declare_io(
            inputs={"V":1, "tau_load":1}, 
            outputs={"i":1, "omega":1}, 
            state_size=2)
        for n,v in {"R":R,"L":L,"ke":ke,"kt":kt,"J":J,"b":b}.items():
            self.declare_param(n, v)
        self.state_alias = {"i":0, "omega":1}

    def ode(self, state, inputs, t):
        i     = state[:, 0:1]
        omega = state[:, 1:2]
        V     = inputs.get("V", torch.zeros_like(i))
        tau   = inputs.get("tau_load", torch.zeros_like(i))
        R,L,ke,kt,J,b = (self.get_param(n) for n in ("R","L","ke","kt","J","b"))

        di     = (V - R*i - ke*omega) / L
        domega = (kt*i - b*omega - tau) / J

        dx = torch.cat([di, domega], dim=1)
        outs = {"i": i, "omega": omega}
        return dx, outs
