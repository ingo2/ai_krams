import viz_tools
from engine import Value
from matplotlib import pyplot as plt


def main():
    h = 0.0001

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f
    L.label = "L"
    L.backward()

    graph = viz_tools.draw_dot(L)
    graph.view()


if __name__ == "__main__":
    main()
