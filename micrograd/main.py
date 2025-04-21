import random
import sys
import viz_tools
from engine import Value
from matplotlib import pyplot as plt
from neural_net import MLP
from typing import List


def main() -> None:
    random.seed(657)

    # Possible inputs.
    # fmt: off
    xs: List[List[float]] = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5], 
        [0.5, 1.0, 1.0], 
        [1.0, 1.0, -1.0]
    ]
    # fmt: on
    # Desired targets.
    ys: List[float] = [1.0, -1.0, -1.0, 1.0]
    # Initialize the MLP.
    n: MLP = MLP(3, [4, 4, 1])

    loss: Value = Value(sys.float_info.max)
    loss_prev: float = loss.data
    for k in range(10):
        ypred: List[Value] = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        n.zero_grad()
        loss.backward()

        for p in n.parameters():
            p.data += -0.01 * p.grad

        decr: float = loss_prev - loss.data
        loss_prev = loss.data

        print(f"{k}: loss = {loss.data}, decr = {decr}")

    graph = viz_tools.draw_dot(loss)
    graph.view()


if __name__ == "__main__":
    main()
