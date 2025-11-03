"""
Autograd engine for symbolic gradient calculation.

This module implements automatic differentiation through computational graph tracking.
"""

import math


class Value:
    """
    A Value represents a scalar value in a computational graph.

    It tracks operations performed on it and enables automatic differentiation
    through backpropagation.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        """
        Initialize a Value node.

        Args:
            data: The scalar value (float or int)
            _children: Tuple of Value objects that produced this value
            _op: String describing the operation that produced this value
            label: Optional label for debugging/visualization
        """
        self.data = float(data)
        self.grad = 0.0  # Gradient of the loss with respect to this value
        self._backward = lambda: None  # Function to propagate gradients
        self._prev = set(_children)  # Parent nodes in the computational graph
        self._op = _op  # Operation that produced this node
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """Addition operation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """Multiplication operation."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        """Power operation (only supports int/float exponents)."""
        assert isinstance(other, (int, float)), (
            f"Power operation only supports int or float exponents, got {type(other).__name__}"
        )
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        """Division operation."""
        return self * other**-1

    def __neg__(self):
        """Negation operation."""
        return self * -1

    def __sub__(self, other):
        """Subtraction operation."""
        return self + (-other)

    def __radd__(self, other):
        """Reverse addition (for operations like: 2 + Value(3))."""
        return self + other

    def __rmul__(self, other):
        """Reverse multiplication (for operations like: 2 * Value(3))."""
        return self * other

    def __rsub__(self, other):
        """Reverse subtraction (for operations like: 2 - Value(3))."""
        return other + (-self)

    def __rtruediv__(self, other):
        """Reverse division (for operations like: 2 / Value(3))."""
        return other * self**-1

    def relu(self):
        """ReLU activation function."""
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        """Tanh activation function."""
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        """Sigmoid activation function."""
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        """Exponential function."""
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):
        """Natural logarithm."""
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """
        Compute gradients using backpropagation.

        This performs a topological sort of the computational graph and then
        propagates gradients from this node (assumed to be the output) back
        through all parent nodes.
        """
        # Build topological order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize gradient of output to 1
        self.grad = 1.0

        # Propagate gradients backward through the graph
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = 0.0
