"""
Neural Network building blocks.

This module provides classes for building neural networks using the autograd engine.
"""

import random
from .engine import Value


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """Return a list of all learnable parameters."""
        return []


class Neuron(Module):
    """A single neuron with weights, bias, and optional activation function."""

    def __init__(self, nin, activation="relu"):
        """
        Initialize a neuron.

        Args:
            nin: Number of input features
            activation: Activation function ('relu', 'tanh', 'sigmoid', or None)
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        """
        Forward pass through the neuron.

        Args:
            x: List of input Values

        Returns:
            Output Value after activation
        """
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply activation function
        if self.activation == "relu":
            out = act.relu()
        elif self.activation == "tanh":
            out = act.tanh()
        elif self.activation == "sigmoid":
            out = act.sigmoid()
        else:
            out = act

        return out

    def parameters(self):
        """Return all parameters (weights and bias)."""
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(nin={len(self.w)}, activation={self.activation})"


class Layer(Module):
    """A layer of neurons."""

    def __init__(self, nin, nout, activation="relu"):
        """
        Initialize a layer.

        Args:
            nin: Number of input features
            nout: Number of output neurons
            activation: Activation function for all neurons in the layer
        """
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        """
        Forward pass through the layer.

        Args:
            x: List of input Values

        Returns:
            List of output Values from all neurons
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """Return all parameters from all neurons."""
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Multi-Layer Perceptron (feedforward neural network)."""

    def __init__(self, nin, nouts, activation="relu"):
        """
        Initialize an MLP.

        Args:
            nin: Number of input features
            nouts: List of output sizes for each layer
            activation: Activation function ('relu', 'tanh', 'sigmoid', or None)
                       For the last layer, no activation is applied by default.
        """
        sz = [nin] + nouts
        # All hidden layers use the specified activation
        # Last layer has no activation by default (useful for regression or custom loss)
        self.layers = []
        for i in range(len(nouts)):
            act = activation if i < len(nouts) - 1 else None
            self.layers.append(Layer(sz[i], sz[i + 1], activation=act))

    def __call__(self, x):
        """
        Forward pass through the MLP.

        Args:
            x: List of input Values or a single list/Value

        Returns:
            Output from the final layer
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Return all parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# Loss functions


def mse_loss(predictions, targets):
    """
    Mean Squared Error loss.

    Args:
        predictions: List of predicted Values
        targets: List of target values (can be Values or scalars)

    Returns:
        MSE loss as a Value
    """
    targets = [t if isinstance(t, Value) else Value(t) for t in targets]
    loss = sum((pred - target) ** 2 for pred, target in zip(predictions, targets))
    return loss * (1.0 / len(predictions))


def binary_cross_entropy(predictions, targets):
    """
    Binary Cross Entropy loss.

    Args:
        predictions: List of predicted Values (should be between 0 and 1)
        targets: List of target values (0 or 1, can be Values or scalars)

    Returns:
        BCE loss as a Value
    """
    targets = [t if isinstance(t, Value) else Value(t) for t in targets]
    eps = 1e-7  # Small constant to avoid log(0)

    # Create epsilon Values once for reuse
    eps_value = Value(eps)
    one_value = Value(1.0)

    loss = Value(0.0)
    for pred, target in zip(predictions, targets):
        # Clamp predictions to avoid log(0) but keep them in the computational graph
        # We add a tiny constant instead of creating a new Value each iteration
        safe_pred = pred + eps_value
        one_minus_pred = (one_value - pred) + eps_value

        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        term1 = target * safe_pred.log()
        term2 = (one_value - target) * one_minus_pred.log()
        loss = loss - (term1 + term2)

    return loss * (1.0 / len(predictions))
