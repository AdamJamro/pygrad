from __future__ import annotations

from typing import Callable

from numpy import ndarray
from numpy import array
from numpy import ones

_name_id = 0
_identity_fn = lambda x: x
def free_name():
    global _name_id
    name = f"var_{_name_id}"
    _name_id += 1
    return name




class Variable:
    __slots__ = ['name', 'value', 'local_gradients']

    def __init__(self, value=None, name: str=None, local_gradients: tuple[tuple[Variable, Callable], ...]=None):
        self.name = name if name is not None else free_name()
        self.value = value if isinstance(value, ndarray) else array(value)
        self.local_gradients = local_gradients if local_gradients is not None else []

    @staticmethod
    def constant(value, name: str=free_name()):
        return Variable(value=value, name=name)


    def __repr__(self):
        return f"Variable(value={self.value})"

    # operators have no explicit type checks for micro optimization as static typechecks should be enough
    def __add__(self, other: Variable):
        return operator_add(self, other)

# help with redundant constants (another micro optimization)
_ONE = Variable.constant(ones((1,)))


def operator_add(var1: Variable, var2: Variable) -> Variable:
        local_gradients: tuple[tuple[Variable, Callable], ...] = (
            (self, _identity_fn,),
            (other, _identity_fn,),
        )
        return Variable(self.value + other.value, local_gradients=list(local_gradients))
