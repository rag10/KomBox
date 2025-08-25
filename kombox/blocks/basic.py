from ..core.block import ContinuousBlock

class Constant(ContinuousBlock):
    def __init__(self, width=1, value=0.0):
        super().__init__()
        self.declare_io(outputs={"out": width})
        self.declare_param("value", value)

    def ode(self, state, inputs, t):
        val = self.get_param("value")
        width = self.out_specs["out"].n
        if val.shape[1] == 1 and width > 1: val = val.expand(-1, width)
        dx = state.new_zeros((state.shape[0], self.state_size()))
        return dx, {"out": val}


class Gain(ContinuousBlock):
    def __init__(self, width=1, gain=1.0):
        super().__init__()
        self.declare_io(inputs={"in": width}, outputs={"out": width})
        self.declare_param("K", gain)

    def ode(self, state, inputs, t):
        u = inputs["in"]
        K = self.get_param("K")
        if K.shape[1] == 1: K = K.expand(-1, u.shape[1])
        dx = state.new_zeros((state.shape[0], self.state_size()))
        return dx, {"out": K * u}
    
class Adder(ContinuousBlock):
    def __init__(self, n_inputs=2, width=1):
        super().__init__()
        self.declare_io(outputs={"out": width},
                        repeated_inputs={"in": (int(n_inputs), int(width))})

    def ode(self, state, inputs, t):
        acc = None
        count = sum(1 for k in self.in_specs if k.startswith("in"))
        for i in range(1, count+1):
            xi = inputs[f"in{i}"]
            acc = xi if acc is None else acc + xi
        dx = state.new_zeros((state.shape[0], self.state_size()))
        return dx, {"out": acc}
    
class Product(ContinuousBlock):
    def __init__(self, n_inputs=2, width=1):
        super().__init__()
        self.declare_io(outputs={"out": width},
                        repeated_inputs={"in": (int(n_inputs), int(width))})

    def ode(self, state, inputs, t):
        result = None
        count = sum(1 for k in self.in_specs if k.startswith("in"))
        
        for i in range(1, count + 1):
            xi = inputs[f"in{i}"]
            result = xi if result is None else result * xi
            
        dx = state.new_zeros((state.shape[0], self.state_size()))
        return dx, {"out": result}