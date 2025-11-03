"""PyGrad - A neural network library with autograd automatic gradient calculation."""

from .autodiff.variable import Variable, clear_tape

__all__ = ["Variable", "clear_tape"]