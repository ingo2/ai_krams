import random
import sys
from engine import Value
from neural_net import MLP
import pytest

# pytest -s
def test_training_loop():
    random.seed(657)

    # Possible inputs.
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    # Desired targets.
    ys = [1.0, -1.0, -1.0, 1.0]
    # Initialze the MLP.
    n = MLP(3, [4, 4, 1])

    # Training loop.
    loss = Value(sys.float_info.max)
    loss_prev = loss.data
    for k in range(10):
        # Compute predictions and loss (forward pass).
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # Reset and recompute gradients (backward pass).
        n.zero_grad()
        loss.backward()

        # Apply gradient decent.
        for p in n.parameters():
            p.data += -0.01 * p.grad

        decr = loss_prev - loss.data
        loss_prev = loss.data

        print(f"{k}: loss = {loss.data}, decr = {decr}")
        assert decr > 0.0

    # Regression check.
    loss.data == pytest.approx(0.23246414201773874, rel=1e-6)
