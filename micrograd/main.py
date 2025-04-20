import random
import sys
import viz_tools
from engine import Value
from matplotlib import pyplot as plt
from neural_net import MLP


def main():
    random.seed(657)

    # Possible inputs.
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    # Desired targets.
    ys = [1.0, -1.0, -1.0, 1.0]
    # Initialze the MLP.
    n = MLP(3, [4, 4, 1])

    loss = Value(sys.float_info.max)
    loss_prev = loss.data
    for k in range(10):
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        n.zero_grad()
        loss.backward()

        for p in n.parameters():
            p.data += -0.01 * p.grad

        decr = loss_prev - loss.data
        loss_prev = loss.data

        print(f"{k}: loss = {loss.data}, decr = {decr}")

    graph = viz_tools.draw_dot(loss)
    graph.view()


if __name__ == "__main__":
    main()
