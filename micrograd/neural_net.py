import random
from typing import List, Union
from engine import Value


class Module:

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):

    def __init__(self, nin: int, nonlin: str = "relu") -> None:
        self.w: List[Value] = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(0)
        self.nonlin: str = nonlin

    def __call__(self, x: List[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == "relu":
            return act.relu()
        elif self.nonlin == "tanh":
            return act.tanh()
        elif self.nonlin == "linear":
            return act
        else:
            raise ValueError(f"Unknown activation function: {self.nonlin}")

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        act = "UnknownActivation"
        if self.nonlin == "relu":
            act = "ReLU"
        elif self.nonlin == "tanh":
            act = "Tanh"
        elif self.nonlin == "linear":
            act = "Linear"

        return f"{act}-Neuron({len(self.w)})"


class Layer(Module):
    """A layer of neurons."""

    def __init__(self, nin: int, nout: int, **kwargs) -> None:
        self.neurons: List[Neuron] = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Multi-layer perceptron."""

    def __init__(self, nin: int, nouts: List[int], nonLin: str) -> None:
        sz = [nin] + nouts
        self.layers: List[Layer] = [
            Layer(sz[i], sz[i + 1], nonlin=nonLin if (i != len(nouts) - 1) else "linear")
            for i in range(len(nouts))
        ]

    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
