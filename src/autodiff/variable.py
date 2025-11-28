"""
@Author Adam Jamroziński


Variable is the building block of the computation graph
each overloaded or custom operator saves itself as a record on a tape stack

The _tape_stack is bundled with the variable module by default.
It contains all the information about which variable was needed to compute the other.
The information how to compute the backward partial derivative at each step.
Thanks to this approach, the list is being automatically sorted topologically (in reverse order)

Each variable in the computation graph can be passed to the grad(...) function to compute its derivative
"""

from __future__ import annotations

import functools
from functools import lru_cache
from typing import Iterable, Sequence, TypeAlias, Any, Callable

from numpy import ndarray, array, ones, shape, einsum, pad, ones_like
from numpy.lib._stride_tricks_impl import as_strided

from src.autodiff.tape import Tape


TYPE_SAFE = True
# import _tape_stack only if you know how to properly use the Tape(...) and stack
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
    __slots__ = ("name", "value")

    def __init__(self, value: numeric = None, name: str = None):
        self.name = name or free_name()
        self.value = value if isinstance(value, ndarray) else array(value)

    # TODO how to memoize everything by default?
    @staticmethod
    @lru_cache(None)
    def constant(value: numeric, name: str = None):
        """micro-optimization memoizing constant variables since they don't lead to cycles in the computation graph"""
        return Variable(value=value, name=name)

    @staticmethod
    def convert_input_into_variable(
        operator_impl,
    ) -> Callable[[Any, ...], tuple[Variable, ...]]:
        @functools.wraps(operator_impl)
        def operator_wrapped(self, *raw_others: numeric | Variable):
            if not TYPE_SAFE:
                return operator_impl(self, *raw_others)
            converted: Variable
            converted_args: list[Variable] = []
            for raw_other in raw_others:
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
                converted_args.append(converted)
            return operator_impl(self, *converted_args)

        return operator_wrapped

    def __repr__(self):
        return f"{self.name}(value={self.value})"

    # operators have no explicit type checks for micro optimization as static typechecks should be enough
    @convert_input_into_variable
    def __add__(self, other: Variable):
        return operator_add(self, other)

    @convert_input_into_variable
    def __radd__(self, other: Variable):
        return operator_add(self, other)

    @convert_input_into_variable
    def __neg__(self, other: Variable):
        return operator_neg(self)

    @convert_input_into_variable
    def __sub__(self, other: Variable):
        return operator_add(self, -other)

    @convert_input_into_variable
    def __rsub__(self, other: Variable):
        return operator_add(-other, self)

    @convert_input_into_variable
    def __mul__(self, other: Variable):
        return operator_mul(self, other)

    @convert_input_into_variable
    def __rmul__(self, other: Variable):
        return operator_mul(self, other)

    @convert_input_into_variable
    def __matmul__(self, other: Variable):
        return operator_matmul(self, other)

    @convert_input_into_variable
    def __rmatmul__(self, other: Variable):
        return operator_matmul(other, self)

    @convert_input_into_variable
    def convolve(self, kernel: Variable):
        """Convolve 2d 2D o padding"""
        if TYPE_SAFE:
            assert self.value.ndim == 2 and kernel.value.ndim == 2, (
                "2D convolution requires both input and kernel to be 2D arrays."
            )
            assert (
                self.value.shape[0] >= kernel.value.shape[0]
                and self.value.shape[1] >= kernel.value.shape[1]
            ), (
                "Kernel size must be smaller than or equal to input size for convolution."
            )

        return convolution_2d(self, kernel)

    @convert_input_into_variable
    def ReLU(self):
        return operator_relu(self)


    def backward(self, directional_grad: Variable | None = None) -> dict[Variable, Variable]:
        """Convenience method to compute gradients of this variable with respect to all other variables in the graph."""
        return grad(self, directional_grad=directional_grad)


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
        dLoss_dVar1, dLoss_dVar2 = dLoss_dAdditionResult, dLoss_dAdditionResult
        dLoss_dInputs = (dLoss_dVar1, dLoss_dVar2)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_neg(var: Variable) -> Variable:
    forward = Variable(-var.value)

    inputs = (var,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dNegationResult,) = dLoss_dOutputs
        dLoss_dVar = -dLoss_dNegationResult
        dLoss_dInputs = (dLoss_dVar,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_mul(var1: Variable, var2: Variable) -> Variable:
    forward = Variable(var1.value * var2.value)
    # TODO optimize edge case [1,1,...], [0,...] multiplication?

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


def operator_matmul(matrix: Variable, vector: Variable) -> Variable:
    forward = Variable(matrix.value @ vector.value)
    inputs = (matrix, vector)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dMatmulResult,) = dLoss_dOutputs
        if TYPE_SAFE:
            assert dLoss_dMatmulResult.value.shape == forward.value.shape, (
                f"Shape mismatch in matmul backward pass: "
                f"dLoss_dMatmulResult shape {dLoss_dMatmulResult.value.shape} != forward shape {forward.value.shape}"
            )

        dLoss_dMatrix = dLoss_dMatmulResult @ Variable(vector.value.T)
        # maybe swap the order of matmul to get rid of transpose?
        dLoss_dVector = Variable(matrix.value.T) @ dLoss_dMatmulResult
        dLoss_dInputs = (dLoss_dMatrix, dLoss_dVector)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def _image2windows(image: ndarray, window_shape: shape) -> ndarray:
    """
    Extracts a view of sliding windows from the input image based on windws_shape.

    Assumes no padding and up/down sliding stride of 1.
    """
    i_rows, i_cols = image.shape
    w_rows, w_cols = window_shape
    output_rows = i_rows - w_rows + 1
    output_cols = i_cols - w_cols + 1

    sliding_windows_shape = (output_rows, output_cols, w_rows, w_cols)
    sliding_windows_strides = (
        image.strides[0],
        image.strides[1],
        image.strides[0],
        image.strides[1],
    )

    return as_strided(
        image, shape=sliding_windows_shape, strides=sliding_windows_strides
    )


def convolution_2d_forward(matrix: ndarray, kernel: ndarray) -> ndarray:
    """Performs a 2D convolution operation on the input matrix with the given kernel."""
    m_rows, m_cols = matrix.shape
    k_rows, k_cols = kernel.shape
    output_rows, output_cols = m_rows - k_rows + 1, m_cols - k_cols + 1
    output = ndarray((output_rows, output_cols))

    windows = _image2windows(matrix, kernel.shape)
    kernel_flipped = kernel[::-1, ::-1]

    # broadcast multiplication and sum over the last two axes
    return einsum("ijkl,kl->ij", windows, kernel_flipped)


def convolution_2d(matrix: Variable, kernel: Variable):
    forward = Variable(convolution_2d_forward(matrix.value, kernel.value))
    inputs = (matrix, kernel)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dConvResult,) = dLoss_dOutputs

        # Gradient w.r.t. input matrix
        # magic: convolution between padded dLoss_dConvResult and kernel
        kernel_flipped = Variable(kernel.value[::-1, ::-1])
        pad_height = kernel.value.shape[0] - 1
        pad_width = kernel.value.shape[1] - 1
        padded_conv_result = Variable(
            pad(
                dLoss_dConvResult.value,
                ((pad_height, pad_height), (pad_width, pad_width)),
            )
        )
        dLoss_dMatrix = padded_conv_result.convolve(kernel_flipped)

        # Gradient w.r.t. kernel
        # magic: convolution between input matrix and flipped dLoss_dConvResult
        conv_result_flipped = Variable(dLoss_dConvResult.value[::-1, ::-1])
        dLoss_dKernel = matrix.convolve(conv_result_flipped)

        dLoss_dInputs = (dLoss_dMatrix, dLoss_dKernel)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_relu(var: Variable):
    raise NotImplementedError


def grad(
    loss_variable: Variable,
    *,
    tape_records=None,
    directional_grad: Variable | None = None,
    # desired_results: Sequence[Variable] | None = None,
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
    dLoss_d: dict[Variable, Variable] = {
        loss_variable: Variable(
            ones_like(loss_variable.value)
        ) if directional_grad is None else directional_grad
    }

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
