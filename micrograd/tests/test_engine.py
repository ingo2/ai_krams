from engine import Value


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
