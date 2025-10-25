"""
Tests for the autodiff engine.
"""

import math

import pytest

from autodiff import gradient
from autodiff.variable import Variable, clear_tape, get_tape_stack
import autodiff

@pytest.fixture(autouse=True)
def empty_tape_Stack(monkeypatch):
    """Ensure tape is empty before each test."""

    clear_tape()
    yield
    clear_tape()


class TestValueBasicOperations:
    """Test basic arithmetic operations."""

    def test_addition_1st_gradient(self):
        """Test addition and gradient propagation."""
        a = Variable(2.0)
        b = Variable(3.0)
        c = a + b
        dc_d = autodiff.gradient.grad(c)

        assert c.value == 5.0
        assert dc_d[b].value == 1.0
        assert dc_d[a].value == 1.0
        assert dc_d[c].value == 1.0

    def test_addition_2nd_gradient(self):
        """Test addition and gradient propagation."""
        a = Variable(2.0)
        b = Variable(3.0)
        c = a + b

        dc_d = autodiff.gradient.grad(c, get_tape_stack())
        ddcda_d = autodiff.gradient.grad(dc_d[a], get_tape_stack())

        assert c.value == 5.0
        assert dc_d[b].value == 1.0
        assert dc_d[a].value == 1.0
        assert dc_d[c].value == 1.0
        assert ddcda_d[a].value == 0.0

    def test_multiplication(self):
        """Test multiplication and gradient propagation."""
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()

        assert c.data == 6.0
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_power(self):
        """Test power operation and gradient propagation."""
        a = Value(2.0)
        b = a ** 3
        b.backward()

        assert b.data == 8.0
        assert a.grad == 12.0  # d/dx(x^3) = 3x^2 = 3*4 = 12

    def test_division(self):
        """Test division operation."""
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        c.backward()

        assert c.data == 3.0
        assert abs(a.grad - 0.5) < 1e-6
        assert abs(b.grad - (-1.5)) < 1e-6

    def test_subtraction(self):
        """Test subtraction operation."""
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        c.backward()

        assert c.data == 2.0
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_negation(self):
        """Test negation operation."""
        a = Value(5.0)
        b = -a
        b.backward()

        assert b.data == -5.0
        assert a.grad == -1.0


class TestValueReverseOperations:
    """Test reverse operations (e.g., 2 + Value(3))."""

    def test_radd(self):
        """Test reverse addition."""
        a = Value(3.0)
        b = 2.0 + a
        b.backward()

        assert b.data == 5.0
        assert a.grad == 1.0

    def test_rmul(self):
        """Test reverse multiplication."""
        a = Value(3.0)
        b = 2.0 * a
        b.backward()

        assert b.data == 6.0
        assert a.grad == 2.0

    def test_rsub(self):
        """Test reverse subtraction."""
        a = Value(3.0)
        b = 5.0 - a
        b.backward()

        assert b.data == 2.0
        assert a.grad == -1.0

    def test_rtruediv(self):
        """Test reverse division."""
        a = Value(2.0)
        b = 6.0 / a
        b.backward()

        assert b.data == 3.0
        assert abs(a.grad - (-1.5)) < 1e-6


class TestActivationFunctions:
    """Test activation functions."""

    def test_relu_positive(self):
        """Test ReLU with positive input."""
        a = Value(2.0)
        b = a.relu()
        b.backward()

        assert b.data == 2.0
        assert a.grad == 1.0

    def test_relu_negative(self):
        """Test ReLU with negative input."""
        a = Value(-2.0)
        b = a.relu()
        b.backward()

        assert b.data == 0.0
        assert a.grad == 0.0

    def test_tanh(self):
        """Test tanh activation."""
        a = Value(0.5)
        b = a.tanh()
        b.backward()

        expected = math.tanh(0.5)
        assert abs(b.data - expected) < 1e-6
        # Gradient of tanh is 1 - tanh^2
        expected_grad = 1 - expected ** 2
        assert abs(a.grad - expected_grad) < 1e-6

    def test_sigmoid(self):
        """Test sigmoid activation."""
        a = Value(0.5)
        b = a.sigmoid()
        b.backward()

        expected = 1 / (1 + math.exp(-0.5))
        assert abs(b.data - expected) < 1e-6
        # Gradient of sigmoid is sigmoid * (1 - sigmoid)
        expected_grad = expected * (1 - expected)
        assert abs(a.grad - expected_grad) < 1e-6

    def test_exp(self):
        """Test exponential function."""
        a = Value(2.0)
        b = a.exp()
        b.backward()

        expected = math.exp(2.0)
        assert abs(b.data - expected) < 1e-6
        assert abs(a.grad - expected) < 1e-6

    def test_log(self):
        """Test natural logarithm."""
        a = Value(2.0)
        b = a.log()
        b.backward()

        expected = math.log(2.0)
        assert abs(b.data - expected) < 1e-6
        assert abs(a.grad - 0.5) < 1e-6  # d/dx(log(x)) = 1/x = 1/2


class TestComplexComputations:
    """Test more complex computational graphs."""

    def test_chain_rule(self):
        """Test chain rule with nested operations."""
        x = Value(2.0)
        y = x * 2 + 3
        z = y ** 2
        z.backward()

        # z = (2x + 3)^2 = (2*2 + 3)^2 = 7^2 = 49
        assert z.data == 49.0
        # dz/dx = 2 * (2x + 3) * 2 = 4 * 7 = 28
        assert x.grad == 28.0

    def test_multiple_paths(self):
        """Test gradient accumulation through multiple paths."""
        x = Value(3.0)
        y = x * x  # y = x^2
        z = y + x  # z = x^2 + x
        z.backward()

        # z = x^2 + x = 9 + 3 = 12
        assert z.data == 12.0
        # dz/dx = 2x + 1 = 6 + 1 = 7
        assert x.grad == 7.0

    def test_complex_expression(self):
        """Test a more complex expression."""
        a = Value(2.0)
        b = Value(3.0)
        c = Value(-1.0)

        # f = (a * b + c) * (a + b * c)
        d = a * b + c
        e = a + b * c
        f = d * e
        f.backward()

        # d = 2*3 + (-1) = 5
        # e = 2 + 3*(-1) = -1
        # f = 5 * (-1) = -5
        assert f.data == -5.0

        # df/da = b * e + d * 1 = 3*(-1) + 5*1 = 2
        assert a.grad == 2.0
        # df/db = a * e + d * c = 2*(-1) + 5*(-1) = -7
        assert b.grad == -7.0
        # df/dc = 1 * e + d * b = 1*(-1) + 5*3 = 14
        assert c.grad == 14.0

    def test_zero_grad(self):
        """Test gradient zeroing."""
        a = Value(2.0)
        b = a * 3
        b.backward()

        assert a.grad == 3.0

        a.zero_grad()
        assert a.grad == 0.0


class TestBackwardPropagation:
    """Test backward propagation mechanics."""

    def test_multiple_backward_calls(self):
        """Test that multiple backward calls accumulate gradients."""
        x = Value(2.0)
        y = x * 3
        y.backward()

        first_grad = x.grad
        assert first_grad == 3.0

        # Call backward again - gradients should accumulate
        y.backward()
        assert x.grad == first_grad + 3.0  # 6.0

    def test_topological_sort(self):
        """Test that topological sort works correctly."""
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        d = c * 2
        e = d + a
        e.backward()

        # This tests that gradients flow correctly through the graph
        assert e.data == 12.0  # ((2+3)*2) + 2 = 12
        # de/da includes both direct path and path through c,d
        # de/da = 1 (direct) + 2 (through c and d)
        assert a.grad == 3.0
