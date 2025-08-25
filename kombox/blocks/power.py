# kombox/blocks/power.py
from __future__ import annotations
import torch
from kombox.core.block import ContinuousBlock

class BuckAveraged(ContinuousBlock):
    """
    Convertidor Buck modelo promedio:
      dIL/dt   = (D*Vin - Vout)/L
      dVout/dt = (IL - Vout/R)/C
    Estado: [IL, Vout]; Entradas: Vin, D (0..1), R (opcional)
    """
    def __init__(self, L=100e-6, C=100e-6, Rload=10.0):
        super().__init__()
        self.declare_io(inputs={"Vin":1, "D":1, "R":1}, outputs={"IL":1, "Vout":1}, state_size=2)
        self.declare_param("L", L); self.declare_param("C", C); self.declare_param("Rload", Rload)
        self.state_alias = {"IL":0, "Vout":1}

    def ode(self, state, inputs, t):
        IL   = state[:, 0:1]
        Vout = state[:, 1:2]
        Vin  = inputs.get("Vin", torch.zeros_like(IL))
        D    = inputs.get("D", torch.zeros_like(IL)).clamp(0.0, 1.0)
        Rext = inputs.get("R", None)

        L = self.get_param("L"); C = self.get_param("C")
        R = Rext if Rext is not None else self.get_param("Rload")

        dIL   = (D*Vin - Vout) / L
        dVout = (IL - Vout/R) / C

        dx = torch.cat([dIL, dVout], dim=1)
        return dx, {"IL": IL, "Vout": Vout}
