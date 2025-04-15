from engine import Value
from neural_net import Neuron
import pytest
import viz_tools


def test_neuron():
    # No assertions. Just runs the thing.
    n = Neuron(2)
    out = n([1, 2])
    out.backward()

    # Uncommend to optionally draw the graph.
    # graph = viz_tools.draw_dot(out)
    # graph.view()
