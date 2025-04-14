from engine import Value
import pytest
import viz_tools


def test_value():
    a = Value(2.0)
    assert a.data == 2.0
    assert a.grad == 0
    assert a._prev == set()
    assert a._op == ""
    assert repr(a) == "Value(: data=2.0, grad=0)"


def test_value_repr():
    a = Value(5.0)
    assert repr(a) == "Value(: data=5.0, grad=0)"


def test_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c._op == "+"
    assert c.data == 5.0
    assert c._prev == {a, b}


def test_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c._op == "*"
    assert c.data == 6.0
    assert c._prev == {a, b}


def test_add_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(-4.0)
    d = a + b * c
    assert d._op == "+"
    assert d.data == 2.0 + 3.0 * -4.0
    assert len(d._prev) == 2
    assert {child.data for child in d._prev} == {2.0, -12.0}


def test_mul_add():
    a = Value(2.0)
    b = Value(3.0)
    c = Value(-5.0)
    d = a * b + c
    assert d._op == "+"
    assert d.data == 2.0 * 3.0 + -5.0
    assert len(d._prev) == 2
    assert {child.data for child in d._prev} == {-5.0, 6.0}


def test_pow():
    a = Value(2.0)
    b = a**3
    assert b._op == "**3"
    assert b.data == 8.0
    assert len(b._prev) == 1
    assert b._prev == {a}


def test_example1():
    # x1 \
    #     * x1w1 \
    # w1 /        \
    #              + x1w1+x2w2 \
    # x2 \        /             + n:=x1w1+x2w2+b -> tanh(n) -> L
    #     * x2w2 /           b /
    # w2 /
    x1 = Value(2.0, label="x1")
    w1 = Value(-3.0, label="w1")
    x2 = Value(0.0, label="x2")
    w2 = Value(1.0, label="w2")
    x1w1 = x1 * w1
    x1w1.label = "x1w1"
    x2w2 = x2 * w2
    x2w2.label = "x2w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1w1+x2w2"
    b = Value(6.881373587, label="b")
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"

    # Initialize gradient of "o": (o - (o + h)) / h = 1
    o.grad = 1.0
    # Semi-automatic backpropagation: Compute gradients of all nodes in the 
    # graph starting from the output node "o" and going backwards to the input 
    # nodes.
    o._backward()
    n._backward()
    x1w1x2w2._backward()
    x1w1._backward()
    x2w2._backward()

    # graph = viz_tools.draw_dot(o)
    # graph.view()

    assert -1.5 == pytest.approx(x1.grad, rel=1e-4)
    assert 1.0 == pytest.approx(w1.grad, rel=1e-4)
    assert 0.5 == pytest.approx(x2.grad, rel=1e-4)
    assert 0.0 == pytest.approx(w2.grad, rel=1e-4)

    assert 0.5 == pytest.approx(x1w1.grad, rel=1e-4)
    assert 0.5 == pytest.approx(x2w2.grad, rel=1e-4)
    assert 0.5 == pytest.approx(b.grad, rel=1e-4)

    assert 0.5 == pytest.approx(n.grad, rel=1e-4)
