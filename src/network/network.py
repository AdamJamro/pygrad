import abc
from abc import abstractmethod
from itertools import chain
from typing import Sequence, Iterable

from numpy.random import random

from autodiff.variable import Variable, clear_tape, loss_mse, loss_NLL, _tape_stack


class Network(abc.ABC):
    """
    Modular neural network base class.

    To use, inherit from Network e.g.:
    FooNN(Network):
        __init__(self, ...):
            super().__init__()
            declared here fields either of type Variable or Network
            will be stored marked as learnable parameters
            ...
        forward(self, ...):
            ...
            overwrite of abstract forward is being required
            this function needs to return the output Variable
    """

    __slots__ = ("__dict__", "_parameters", "_subnetworks")

    def __init__(self):
        object.__setattr__(self, "_parameters", [])
        object.__setattr__(self, "_subnetworks", [])

    def __setattr__(self, name, value):
        parameters = getattr(self, "_parameters", None)
        subnetworks = getattr(self, "_subnetworks", None)

        if parameters is None or subnetworks is None:
            raise AttributeError(
                f"Cannot assign attributes to Network before {self.__class__.__name__}.__init__() initialization."
            )

        match value:
            case Variable():
                parameters.append(value)
                if name in subnetworks:
                    del subnetworks[name]
            case Network():
                subnetworks.append(value)
                if name in parameters:
                    del parameters[name]

        object.__setattr__(self, name, value)

    @abstractmethod
    def forward(self, *x: Variable) -> Sequence[Variable] | Variable:
        raise NotImplementedError

    def __call__(self, *x: Variable) -> Sequence[Variable] | Variable:
        return self.forward(*x)

    def parameters(self, recurse=True) -> Iterable[Variable]:
        return chain(
            self._parameters,
            (
                param
                for subnetwork in self._subnetworks
                for param in subnetwork.parameters(recurse=recurse)
            ),
        )

    def subnetworks(self, recurse=True) -> Iterable['Network']:
        return chain(
            self._subnetworks,
            (
                sub_subnetwork
                for subnetwork in self._subnetworks
                for sub_subnetwork in subnetwork.subnetworks(recurse=recurse)
            ),
        )


class Linear(Network):
    """
    Ready to use linear subnetwork implementation
    equivalent to Wx+b linear step
    """
    def __init__(self, in_features_length: int, out_features_length: int):
        super().__init__()
        self.weight = Variable(random((out_features_length, in_features_length)))
        self.bias = Variable(
            random(
                (out_features_length, 1)
            )
        )

    def forward(self, x: Variable) -> Variable:
        # todo reduce 2 operations into one by special operator_lin_step(w, b, x)
        return self.weight @ x + self.bias


class Conv2d(Network):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel = Variable(random((kernel_size, kernel_size)))

    def forward(self, x: Variable) -> Variable:
        return Variable.convolve(x, kernel=self.kernel)


class Conv(Network):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel = Variable(random((kernel_size,)))

    def forward(self, x: Variable) -> Variable:
        return Variable.convolve(x, kernel=self.kernel)