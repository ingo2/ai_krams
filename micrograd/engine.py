import math


class Value:
    """See https://github.com/karpathy/micrograd/tree/master"""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0
        # Internal variables used for autograd graph construction.
        self._prev = set(_children)
        # The op that produced this node, for graphviz / debugging / etc.
        self._op = _op
        # A label for the node, for graphviz / debugging / etc.
        self.label = label
        # A backward function that computes the gradient.
        self._backward = lambda: None

    def __repr__(self):
        return f"Value({self.label}: data={self.data}, grad={self.grad})"

    def backward(self):
        """Backpropagation through the graph."""
        topo = topo_sort(self)
        # Initialize gradient: h -> 0 => (F - (F + h)) / h = 1,
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def backward():
            self.grad += other * out.grad ** (other - 1) * out.grad

        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def backward():
            self.grad += out.data * out.grad

        out._backward = backward
        return out

    def tanh(self):
        """Tanh activation function."""
        # Tanh function: f(x) = (2 / (1 + exp(-2x))) - 1
        e2x = math.exp(2 * self.data)
        t = (e2x - 1) / (e2x + 1)
        out = Value(t, (self,), "tanh")

        def backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1


def topo_sort(root):
    """Topological sort of the graph."""
    visited = set()
    order = []

    def dfs(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                dfs(child)
            order.append(node)

    dfs(root)
    return order
