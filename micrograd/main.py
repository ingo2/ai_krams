import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons, make_blobs

from engine import Value
from neural_net import MLP


def loss(X: np.ndarray, y: np.ndarray, model: MLP) -> Value:
    Xb, yb = X, y
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # Forward the model to get scores.
    scores = list(map(model, inputs))

    # SVM "max-margin" loss.
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization.
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


def main() -> None:
    random.seed(1337)
    np.random.seed(1337)

    # Generate dataset.
    X: np.ndarray
    y: np.ndarray
    X, y = make_moons(n_samples=100, noise=0.1)

    y = y * 2 - 1  # map make y be -1 or 1 instead of 0 or 1

    # Visualize in 2D.
    # plt.figure(figsize=(5, 5))
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
    # plt.show()

    # Initialize a model.
    model: MLP = MLP(2, [16, 16, 1])  # 2-layer neural network
    print(model)
    print("Number of parameters: ", len(model.parameters()))

    total_loss, acc = loss(X, y, model)
    print(f"Initial loss: {total_loss}, accuracy: {acc:.4f}")

    for k in range(100):
        # forward
        total_loss, acc = loss(X, y, model)

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (sgd)
        learning_rate = 1.0 - 0.9 * k / 100
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 1 == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

    # Visualize decision boundary.
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


if __name__ == "__main__":
    main()
