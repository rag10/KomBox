from ..core.block import ContinuousBlock
import torch

class Spring(ContinuousBlock):
    def __init__(self, k=10.0, x0=0.0):
        super().__init__()
        self.declare_io(inputs={"x":1}, outputs={"F":1})
        self.declare_param("k", k); self.declare_param("x0", x0)

    def ode(self, state, inputs, t):
        x = inputs["x"]; k = self.get_param("k"); x0 = self.get_param("x0")
        F = -k * (x - x0)
        dx = state.new_zeros((state.shape[0], self.state_size()))
        return dx, {"F": F}

class Damper(ContinuousBlock):
    def __init__(self, c=1.0):
        super().__init__()
        self.declare_io(inputs={"v":1}, outputs={"F":1})
        self.declare_param("c", c)

    def ode(self, state, inputs, t):
        v = inputs["v"]; c = self.get_param("c")
        F = -c * v
        dx = state.new_zeros((state.shape[0], self.state_size()))
        return dx, {"F": F}

class Mass1D(ContinuousBlock):
    def __init__(self, m=1.0):
        super().__init__()
        self.declare_io(inputs={"F":1}, outputs={"x":1,"v":1}, state_size=2)
        self.declare_param("m", m)
        self.state_alias = {"x":0, "v":1}

    def ode(self, state, inputs, t):
        x = state[:, [0]]; v = state[:, [1]]
        F = inputs["F"]; m = self.get_param("m")
        dx = torch.zeros_like(state)
        dx[:, [0]] = v
        dx[:, [1]] = F / m
        return dx, {"x": x, "v": v}
