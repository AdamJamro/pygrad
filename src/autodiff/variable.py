"""
@Author Adam Jamroziński


Variable is the building block of the computation graph
each overloaded or custom operator saves itself as a record on a tape stack

The _tape_stack is bundled with the variable module by default.
It contains all the information about which variable was needed to compute the other.
and the information how to compute the backward partial derivative at each step.
Thanks to this approach, the list is being automatically sorted topologically (in reverse order)

each variable in the computation graph can be passed to the grad(...) function to compute its derivative
"""
from __future__ import annotations

import functools
from functools import lru_cache
from typing import Iterable, Sequence, TypeAlias

from numpy import ndarray, array, ones

from src.autodiff.tape import Tape


TYPE_SAFE = True
_tape_stack: list[Tape] = []


def clear_tape() -> None:
    global _tape_stack
    _tape_stack = []


def get_tape_stack_snapshot() -> list[Tape]:
    """returns a snapshot shallow copy of the tape stack"""
    global _tape_stack
    return _tape_stack[:]


def get_tape_stack() -> list[Tape]:
    global _tape_stack
    return _tape_stack


def roll_new_tape(tape_ref: list[Tape]) -> None:
    global _tape_stack
    _tape_stack = tape_ref


_name_id = 0
_identity_fn = lambda x: x


def free_name():
    global _name_id
    name = f"__var_{_name_id}"
    _name_id += 1
    return name


def reset_name_ids():
    global _name_id
    _name_id = 0

numeric: TypeAlias = int | float | ndarray


class Variable:
    __slots__ = ["name", "value"]

    def __init__(self, value: numeric =None, name: str = None):
        self.name = name or free_name()
        self.value = value if isinstance(value, ndarray) else array(value)

    # TODO how to memoize everything by default?
    @staticmethod
    @lru_cache(None)
    def constant(value: numeric, name: str = None):
        """micro-optimization memoizing constant variables since they don't lead to cycles in the computation graph"""
        return Variable(value=value, name=name)

    @staticmethod
    def convert_input_into_variable(operator_impl) -> Variable:
        @functools.wraps(operator_impl)
        def operator_wrapped(self, raw_other: numeric | Variable):
            if not TYPE_SAFE:
                return operator_impl(self, raw_other)
            converted: Variable
            match raw_other:
                case ndarr if isinstance(ndarr, ndarray):
                    # TODO check supported dimensions
                    converted = Variable(raw_other)
                case raw_numeric if isinstance(raw_numeric, (int, float)):
                    converted = Variable.constant(raw_numeric)
                case already_variable if isinstance(already_variable, Variable):
                    converted = already_variable
                case _:
                    raise TypeError(
                        f"Unsupported type for Variable operation: {type(raw_other).__name__}"
                    )
            return operator_impl(self, converted)

        return operator_wrapped

    def __repr__(self):
        return f"{self.name}(value={self.value})"

    # operators have no explicit type checks for micro optimization as static typechecks should be enough
    @convert_input_into_variable
    def __add__(self, other: Variable):
        return operator_add(self, other)

    @convert_input_into_variable
    def __mul__(self, other: Variable):
        return operator_mul(self, other)


@lru_cache(maxsize=None)
def _ONES(arr_shape: int | Iterable[int], dtype=float) -> Variable:
    return Variable(ones(arr_shape, dtype=dtype))

# some operators have only one result variable so they assume the input was never None

def operator_add(var1: Variable, var2: Variable) -> Variable:
    forward = Variable(var1.value + var2.value)

    inputs = (
        var1,
        var2,
    )
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dAdditionResult,) = dLoss_dOutputs
        dLoss_dvar1, dLoss_dvar2 = dLoss_dAdditionResult, dLoss_dAdditionResult
        dLoss_dInputs = (dLoss_dvar1, dLoss_dvar2)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_mul(var1: Variable, var2: Variable) -> Variable:
    forward = Variable(var1.value * var2.value)
    # TODO optimize edge case 1, 0 multiplication?

    inputs = (
        var1,
        var2,
    )
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dMultiplicationResult,) = dLoss_dOutputs
        dLoss_dVar1 = dLoss_dMultiplicationResult * var2
        dLoss_dVar2 = dLoss_dMultiplicationResult * var1
        dLoss_dInputs = (dLoss_dVar1, dLoss_dVar2)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def grad(
    loss_variable: Variable,
    tape_records=None,
    desired_results: Sequence[Variable] | None = None,
) -> dict[Variable, Variable]:
    """
    Computes gradients of the loss_variable with respect to each Variable in the computation graph,
    dLoss_d lookup map effectively works out to make dLoss_d[x] equal to dLoss/dx

    :returns
        a dictionary of the dloss_variable/d[key] where key is any other variable used to compute the loss_variable

    loss_variable:
        The top node of the computation graph representing the loss.
        If the loss is not a scalar, it's implicitly set to be an array of ones with the same shape as the loss.

    desired_results:
        # TODO currently serves no optimization
        Selects what we exclude from the gradient result list.
        If desired_results is None, gradients for all variables in the computation graph will be computed.

    If the tape does not contain a variable,
    we consider its gradient None (which brings pruning of unused graoh branches to constant time checks).
    """

    if tape_records is None:
        tape_records = get_tape_stack_snapshot()
    dLoss_d: dict[Variable, Variable] = {loss_variable: Variable(ones(()))}

    def prune_unused_outputs(
        outputs: tuple[Variable, ...],
    ) -> tuple[Variable | None, ...]:
        return tuple(
            (dLoss_d[output] if output in dLoss_d else None) for output in outputs
        )

    for tape_record in reversed(tape_records):
        dLoss_dOutputs = prune_unused_outputs(tape_record.outputs)

        if all(dL_dOutput is None for dL_dOutput in dLoss_dOutputs):
            continue  # prune paths equal to zero vectors

        # perform chain rule back propagation
        dLoss_dInputs = tape_record.back_fn(dLoss_dOutputs)

        # the dag computation can have different shapes than simple MLP's so we actually sum all the gradients
        for tape_input, dL_dInput in zip(tape_record.inputs, dLoss_dInputs):
            # we could have used defaultdict(lambda x:0) but this way we keep the notion of what was actually used
            if tape_input not in dLoss_d:
                dLoss_d[tape_input] = dL_dInput
            else:
                dLoss_d[tape_input] += dL_dInput

    # print some information to understand the values of each intermediate
    # for name, value in dLoss_d.items():
    #     print(f"d{loss_variable.name}_d{name} = {value.name}")

    return dLoss_d
