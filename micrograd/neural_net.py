import random
from engine import Value


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1), label=f"w{i+1}") for i in range(nin)]
        self.bias = Value(random.uniform(-1, 1), label="b")

    def __call__(self, x):
        activation = sum(
            (wi * Value(xi, label=f"x{i+1}") for i, (wi, xi) in enumerate(zip(self.weights, x))),
            self.bias,
        )
        return activation.tanh()
