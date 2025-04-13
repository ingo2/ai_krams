from engine import Value


def test_value():
    a = Value(2.0)
    assert a.data == 2.0
    assert a.grad == 0
    assert a._prev == set()
    assert a._op == ""
    assert repr(a) == "Value(data=2.0, grad=0)"


def test_value_repr():
    a = Value(5.0)
    assert repr(a) == "Value(data=5.0, grad=0)"


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
