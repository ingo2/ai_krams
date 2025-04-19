from engine import Value
import pytest
import viz_tools

# Draw a graph:
# graph = viz_tools.draw_dot(val)
# graph.view()


def test_value():
    a = Value(2.0)
    assert a.data == 2.0
    assert a.grad == 0
    assert a._prev == set()
    assert a._op == ""


def test_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c._op == "+"
    assert c.data == 5.0
    assert c._prev == {a, b}

    c.backward()
    assert c.grad == 1.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c._op == "*"
    assert c.data == 6.0
    assert c._prev == {a, b}

    c.backward()
    assert c.grad == 1.0
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_pow():
    a = Value(2.0)
    b = a**3
    assert b._op == "**3"
    assert b.data == 8.0
    assert len(b._prev) == 1
    assert b._prev == {a}

    b.backward()
    assert b.grad == 1.0
    assert a.grad == 3 * 2 ** (3 - 1)  # 3 * 2^2 = 12.0


def test_add_mul():
    a = Value(2.0, label="a")
    b = Value(3.0, label="b")
    c = Value(-5.0, label="c")
    d = a + b * c
    d.label = "d"
    assert d._op == "+"
    assert d.data == 2.0 + 3.0 * -5.0
    assert len(d._prev) == 2
    assert {child.data for child in d._prev} == {2.0, -15.0}

    d.backward()
    # Uncommend draw the graph.
    # graph = viz_tools.draw_dot(d)
    # graph.view()
    assert d.grad == 1.0
    assert a.grad == 1.0
    assert b.grad == -5.0
    assert c.grad == 3.0


def test_mul_add():
    a = Value(2.0, label="a")
    b = Value(3.0, label="b")
    c = Value(-5.0, label="c")
    d = a * b + c
    d.label = "d"
    assert d._op == "+"
    assert d.data == 2.0 * 3.0 + -5.0
    assert len(d._prev) == 2
    assert {child.data for child in d._prev} == {-5.0, 6.0}

    d.backward()
    # Uncommend draw the graph.
    # graph = viz_tools.draw_dot(d)
    # graph.view()
    assert d.grad == 1.0
    assert c.grad == 1.0
    assert a.grad == 3.0
    assert b.grad == 2.0

def test_mul_pow():
    a = Value(3.0, label="a")
    b = Value(-2.0, label="b")
    c = a * b ** 3
    c.label = "c"
    assert c._op == "*"
    assert c.data == 3.0 * (-2.0) ** 3
    assert len(c._prev) == 2
    assert {child.data for child in c._prev} == {-8.0, 3.0}
    
    c.backward()
    # Uncommend draw the graph.
    # graph = viz_tools.draw_dot(c)
    # graph.view()
    assert c.grad == 1.0
    # dc/da = dc/dc * dc/da = 1.0 * (-2.0) ** 3 = -8.0
    assert a.grad == -8.0
    assert b.grad == 36.0 # 3 * (-2)**2 * 3 = 3 * 4 * 3 = 36.0
   

def test_backward_example1():
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

    o.backward()
    # Uncommend draw the graph.
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


def test_backward_example2():
    a = Value(3.0)
    # b = a + a = 2a => b' = db / da = 2
    b = a + a
    b.backward()
    assert a.data == 3.0
    assert a.grad == 2.0
