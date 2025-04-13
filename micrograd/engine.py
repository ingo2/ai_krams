class Value:
    """See https://github.com/karpathy/micrograd/tree/master"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # Internal variables used for autograd graph construction.
        self._prev = set(_children)
        # The op that produced this node, for graphviz / debugging / etc.
        self._op = _op
      
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        return out
