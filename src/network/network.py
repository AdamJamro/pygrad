import abc
from abc import abstractmethod
from typing import List, Sequence

from numpy.random import random

import autodiff.variable
from autodiff.variable import Variable

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
    __slots__ = ("__dict__", "forward", "backward", "_parameters", "_subnetworks")

    def __init__(self):
        object.__setattr__(self, '_parameters', [])
        object.__setattr__(self, '_subnetworks', [])

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
    def forward(self, x: Sequence[Variable]) -> Sequence[Variable]:
        raise NotImplementedError

    def __call__(self , x: Sequence[Variable]) -> Sequence[Variable]:
        return self.forward(x)



class Linear(Network):
    def __init__(self, in_features_length: int, out_features_length: int):
        super().__init__()
        self.weight = random((out_features_length, in_features_length))
        self.bias = (out_features_length,)

    def forward(self, x: Sequence[Variable]) -> Sequence[Variable]:
        alpha = self.weight * x + self.bias
        return alpha
