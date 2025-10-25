from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import Callable, Iterable, NamedTuple

from numpy import ndarray, array, ones


# from autodiff.backward_function import BackFunction
class Tape(NamedTuple):
    outputs: tuple[Variable, ...]
    inputs: tuple[Variable, ...]
    # back_fn: BackFunction
    back_fn: Callable[[tuple[Variable | None, ...]], tuple[Variable, ...]]


_name_id = 0
_identity_fn = lambda x: x

def free_name():
    global _name_id
    name = f"__var_{_name_id}"
    _name_id += 1
    return name


_tape_stack: list[Tape] = []
def clear_tape():
    global _tape_stack
    _tape_stack = []

def get_tape_stack() -> Sequence[Tape]:
    return _tape_stack


class Variable:
    __slots__ = ['name', 'value', 'local_gradients']

    def __init__(self, value=None, name: str=None):
        self.name = name or free_name()
        self.value = value if isinstance(value, ndarray) else array(value)

    @staticmethod
    @lru_cache(None)
    def constant(value, name: str=free_name()):
        return Variable(value=value, name=name)

    def __repr__(self):
        return f"{self.name}(value={self.value})"

    # operators have no explicit type checks for micro optimization as static typechecks should be enough
    def __add__(self, other: Variable):
        return operator_add(self, other)

    def __mul__(self, other: Variable):
        return operator_mul(self, other)


# help with redundant constants (another micro optimization)
@lru_cache(maxsize=None)
def _ONES(arr_shape: int | Iterable[int], dtype=float) -> Variable:
    return Variable(ones(arr_shape, dtype=dtype))


def operator_add(var1: Variable, var2: Variable) -> Variable:
    forward = Variable(var1.value + var2.value)
    # local_gradients: tuple[tuple[Variable, Callable], ...] = (
    #     (var1, _identity_fn),
    #     (var2, _identity_fn),
    # )

    inputs = (var1, var2,)
    outputs = (forward,)
    def back_fn(dLoss_dOutputs: tuple[Variable, ...]) -> tuple[Variable, ...]:
        dLoss_dAdditionResult, = dLoss_dOutputs
        dLoss_dvar1, dLoss_dvar2 = dLoss_dAdditionResult,  dLoss_dAdditionResult
        dLoss_dInputs = (dLoss_dvar1, dLoss_dvar2)
        return dLoss_dInputs

    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_mul(var1: Variable, var2: Variable) -> Variable:
    forward = Variable(var1.value * var2.value)
    # TODO optimize scalar multiplication?

    inputs = (var1, var2,)
    outputs = (forward,)
    def back_fn(dLoss_dOutputs: tuple[Variable, ...]) -> tuple[Variable, ...]:
        dLoss_dMultiplicationResult, = dLoss_dOutputs
        dLoss_dVar1 = Variable(dLoss_dMultiplicationResult.value * var2.value)
        dLoss_dVar2 = Variable(dLoss_dMultiplicationResult.value * var1.value)
        dLoss_dInputs = (dLoss_dVar1, dLoss_dVar2)
        return dLoss_dInputs

    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward