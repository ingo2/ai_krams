import viz_tools
from engine import Value
from matplotlib import pyplot as plt


def main():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    graph = viz_tools.draw_dot(c)
    graph.view()


if __name__ == "__main__":
    main()
