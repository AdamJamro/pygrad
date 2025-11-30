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
from typing import Iterable, Sequence, TypeAlias, Any, Callable, Literal

from numpy import (
    ndarray,
    array,
    ones,
    shape,
    einsum,
    pad,
    ones_like,
    flip,
    exp,
    zeros,
    tanh,
    maximum,
    log, subtract, square, mean,
)
from numpy.lib._stride_tricks_impl import as_strided, broadcast_to
from numpy.ma.core import allclose

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


def operator_transpose(var: Variable):
    forward = Variable(var.value.T)

    inputs = (var,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dTransposedResult,) = dLoss_dOutputs
        dLoss_dVar = dLoss_dTransposedResult.T
        dLoss_dInputs = (dLoss_dVar,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


class Variable:
    __slots__ = ("name", "value")

    def __init__(self, value: numeric | Iterable[numeric] = None, name: str = None):
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
    ) -> Callable[[Any, ...], Sequence[Variable]]:
        @functools.wraps(operator_impl)
        def operator_wrapped(self, *raw_inputs: numeric | Variable):
            if not TYPE_SAFE:
                return operator_impl(self, *raw_inputs)
            converted_args: list[Variable] = []
            for raw_other in raw_inputs:
                converted: Variable
                match raw_other:
                    case ndarr if isinstance(ndarr, ndarray):
                        # TODO check supported dimensions
                        converted_args.append(Variable(raw_other))
                    case raw_numeric if isinstance(raw_numeric, (int, float)):
                        converted_args.append(Variable.constant(raw_numeric))
                    case already_variable if isinstance(already_variable, Variable):
                        converted_args.append(already_variable)
                    case _:
                        raise TypeError(
                            f"Unsupported type for Variable operation: {type(raw_other).__name__}"
                        )
            return operator_impl(self, *converted_args)

        return operator_wrapped

    def __repr__(self):
        return f"{self.name}(value={self.value})"

    # operators have no explicit type checks for micro optimization as static typechecks should be enough
    @convert_input_into_variable
    def __add__(self, other: Variable | numeric):
        return operator_add(self, other)

    def __radd__(self, other: Variable | numeric):
        return self + other

    @convert_input_into_variable
    def __neg__(self, other: Variable | numeric):
        return operator_neg(self)

    @convert_input_into_variable
    def __sub__(self, other: Variable | numeric):
        return operator_add(self, -other)

    @convert_input_into_variable
    def __rsub__(self, other: Variable | numeric):
        return operator_add(-other, self)

    @convert_input_into_variable
    def __mul__(self, other: Variable | numeric):
        if TYPE_SAFE:
            if self.shape != other.shape:
                other = self.broadcast_to(self.shape)
        return operator_mul(self, other)

    def __rmul__(self, other: Variable | numeric):
        # elementwise multiplication forced commutative
        return self * other

    @convert_input_into_variable
    def __matmul__(self, other: Variable | ndarray):
        if TYPE_SAFE:
            if other.ndim == 1:
                other = other.reshape_as_matrix()
        return operator_matmul(self, other)

    @convert_input_into_variable
    def __rmatmul__(self, other: Variable | ndarray):
        return operator_matmul(other, self)

    def flip(self) -> Variable:
        return operator_flip(self)

    def pad(self, pad_width: Sequence[tuple[int, int]]) -> Variable:
        return operator_pad(self, pad_width=pad_width)

    def sum(self, axis: tuple[int, ...] | int | None = None, keepdims = False) -> Variable:
        return operator_sum(self, axis=axis, keepdims=keepdims)

    def broadcast_to(self, shape) -> Variable:
        return operator_broadcast_to(self, broadcast_shap=shape)

    def reshape(self, shape) -> Variable:
        return operator_reshape(self, new_shape=shape)

    @property
    def shape(self):
        return self.value.shape

    @property
    def T(self) -> Variable:
        return operator_transpose(self)

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    @staticmethod
    def convolve(matrix: Variable, kernel: Variable):
        """Convolve matrix with kernel, both"""
        if TYPE_SAFE:
            assert matrix.value.ndim == kernel.value.ndim, (
                "convolution arguments dimensionality must match."
            )
            assert (
                matrix.value.shape[0] > kernel.value.shape[0]
                and matrix.value.shape[1] > kernel.value.shape[1]
            ), (
                "Kernel size must be smaller than input size for convolution. Use elementwise multiplication instead."
            )

        return operator_convolution(matrix, kernel)

    @staticmethod
    def ReLU(variable: Variable):
        return activation_ReLU(variable)

    @staticmethod
    def sigmoid(variable: Variable):
        return activation_sigmoid(variable)

    @staticmethod
    def tanh(variable: Variable):
        return activation_tanh(variable)

    def backward(
        self, directional_grad: Variable | None = None
    ) -> dict[Variable, Variable]:
        """Convenience method to compute gradients of this variable with respect to all other variables in the graph."""
        global _tape_stack
        return grad(self, directional_grad=directional_grad, tape_records=_tape_stack)


@lru_cache(maxsize=None)
def _ONES(arr_shape: int | Iterable[int], dtype=None) -> Variable:
    """
    Returns cached Variable of ones with a given shape and type

    Thus this should never be a learnable parameter (must stay immutable)
    """
    return Variable(ones(arr_shape, dtype=dtype))


@lru_cache(maxsize=None)
def _ZEROS(arr_shape: int | Iterable[int], dtype=None) -> Variable:
    """
    Returns cached Variable of zeros with a given shape and type

    Thus this should never be a learnable parameter (must stay immutable)
    """
    return Variable(zeros(arr_shape, dtype=dtype))


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


def operator_mul(
    var1: Variable, var2: Variable, peephole_optimization=False
) -> Variable:
    if peephole_optimization:
        if allclose(var1, _ONES(var1.shape)):
            return var2
        if allclose(var2, _ONES(var2.shape)):
            return var1
        if allclose(var1, _ZEROS(var1.shape)) or allclose(var2, _ZEROS(var2.shape)):
            return _ZEROS(var1.shape) * _ZEROS(var2.shape)

    forward = Variable(var1.value * var2.value)
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
    if TYPE_SAFE:
        if vector.ndim == 1:
            raise ValueError("vectors must be explicitly padded with an extra dimension. Reshape as (..., n, 1)")


    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dMatmulResult,) = dLoss_dOutputs

        matrix_T = matrix.T
        dLoss_dVector = matrix_T @ dLoss_dMatmulResult
        match vector.ndim:
            case 1:
                raise ValueError("Forbidden matrix-vector multiplication in backward pass. "
                                 "Explicitly shape vector to (..., n, 1).")
            case 2:
                dLoss_dMatrix = dLoss_dMatmulResult @ vector.T
            case 3:
                # TODO vector.T should be batch aware
                batched_dLoss_dMatrix = dLoss_dMatmulResult @ vector.T
                # remove unnecessary batch dimension
                dLoss_dMatrix = batched_dLoss_dMatrix.sum(axis=tuple(range(vector.ndim - 2)))
            case _:
                raise ValueError("Matrix multiplication supports only 1D and 2D vectors.")

        dLoss_dInputs = (dLoss_dMatrix, dLoss_dVector)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_flip(matrix: Variable, **kwargs) -> Variable:
    """
    flip the array variable along dimension
    equivalent to numpy.flip

    operator_flip(array) is equivalent to array[::-1,::-1, ..., ::-1]
    """
    forward = Variable(flip(matrix.value, **kwargs))

    inputs = (matrix,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dRotatedResult,) = dLoss_dOutputs
        dLoss_dMatrix = dLoss_dRotatedResult.flip()
        dLoss_dInputs = (dLoss_dMatrix,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_pad(matrix: Variable, pad_width: Sequence[tuple[int, int]]) -> Variable:
    """
    Pads the array variable according to pad_width
    pad_width ~ ((before(dim), after(dim)) for dim in each dimension)
    """
    forward = Variable(pad(matrix.value, pad_width=pad_width))

    inputs = (matrix,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dPaddedResult,) = dLoss_dOutputs

        # slices = tuple(
        #     slice(pad_width_dim[0], dLoss_dPaddedResult.value.shape[i] - pad_width_dim[1])
        #     for i, pad_width_dim in enumerate(pad_width)
        # )
        # dLoss_dMatrix = Variable(dLoss_dPaddedResult.value[slices])

        dLoss_dMatrix = matrix  # variables are immutable during grad computation, so only link dL_dIn matrix with dLdOut matrix padded with constants
        dLoss_dInputs = (dLoss_dMatrix,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


# TODO debug
def operator_broadcast_to(tensor: Variable, broadcast_shap) -> Variable:
    """
    Broadcasts the array variable to the given shape
    equivalent to numpy.broadcast_to

    Beware this function uses numpy's view under the hood
    so the result of this operator must remain immutable.
    """
    forward = Variable(broadcast_to(tensor.value, shape=broadcast_shap))

    inputs = (tensor,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dBroadcastedResult,) = dLoss_dOutputs

        # sum over the broadcasted dimensions
        reduce_extra_axes = tuple(
            range(dLoss_dBroadcastedResult.ndim - tensor.value.ndim)
        )
        intermediate_sum = dLoss_dBroadcastedResult.sum(axis=reduce_extra_axes)
        reduce_inner_axes = tuple(
            axis
            for axis, (dim1, dim2) in enumerate(
                zip(tensor.value.shape, intermediate_sum.shape)
            )
            if dim1 != dim2
        )
        dLoss_dTensor = dLoss_dBroadcastedResult.sum(
            axis=reduce_inner_axes, keepdims=True
        )
        if TYPE_SAFE:
            assert dLoss_dTensor.shape == tensor.shape, (
                "shape mismatch in broadcast_to backward pass"
            )

        dLoss_dInputs = (dLoss_dTensor,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_reshape(tensor: Variable, new_shape) -> Variable:
    """
    Reshape tensor using numpy.reshape
    """
    forward = Variable(tensor.value.reshape(new_shape))
    inputs = (tensor,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dResult,) = dLoss_dOutputs
        dLoss_dInput = dLoss_dResult.reshape(tensor.shape)
        dLoss_dInputs = (dLoss_dInput,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def operator_sum(
    tensor: Variable,
    axis: tuple[int, ...] | int | None = None,
    keepdims: bool = False,
) -> Variable:
    """
    Sums the elements of the tensor along the specified axis.
    If axis is None, sums all elements.
    """
    forward = Variable(tensor.value.sum(axis=axis, keepdims=keepdims))

    inputs = (tensor,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dSumResult,) = dLoss_dOutputs
        # without expand_dims this may crash, but it is no bug, we disallow such broadcasting in this version
        dLoss_dTensor = dLoss_dSumResult.broadcast_to(tensor.shape)
        dLoss_dInputs = (dLoss_dTensor,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def _stream2windows(stream: ndarray, window_length: shape) -> ndarray:
    """
    Extracts a view of sliding windows from the input 1D stream based on window_length.
    Assumes no padding and left/right sliding stride of 1.
    """
    (stream_length,) = stream.shape
    output_length = stream_length - window_length + 1

    sliding_windows_shape = (output_length, window_length)
    sliding_windows_strides = (stream.strides[0], stream.strides[0])

    return as_strided(
        stream, shape=sliding_windows_shape, strides=sliding_windows_strides
    )


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


def _space2windows(space: ndarray, window_shape: shape) -> ndarray:
    """
    Extracts a view of sliding windows from the input space based on windws_shape.
    Assumes no padding and sliding stride of 1 in all dimensions.
    """
    s_1d, s_2d, s_3d = space.shape
    w_1d, w_2d, w_3d = window_shape
    output_1d = s_1d - w_1d + 1
    output_2d = s_2d - w_2d + 1
    output_3d = s_3d - w_3d + 1

    sliding_windows_shape = (output_1d, output_2d, output_3d, w_1d, w_2d, w_3d)
    sliding_windows_strides = (
        space.strides[0],
        space.strides[1],
        space.strides[2],
        space.strides[0],
        space.strides[1],
        space.strides[2],
    )

    return as_strided(
        space, shape=sliding_windows_shape, strides=sliding_windows_strides
    )


def convolve_forward(
    tensor: ndarray,
    kernel: ndarray,
    optimize: Literal["greedy", "optimal"] | bool | Sequence | None = True,
) -> ndarray:
    """
    Performs a convolution operation on the input ndarray with the given kernel.

    Matrix and Kernel must adhere to constraints.
    This library version supports 1d,2d and 3d convolutions, but more dimensions are straightforward to implement.
    Note that if the length of dimensions of tensor and kernel match, custom 4D, 5D, and so on sliding windows
    result in the exact same backward functions!

    This implementation does not flip the kernel, effectively performing a Convolution* a.k.a. Cross-Correlation.
    Note we do not perform padding and assume a stride of 1.

    The 'optimize' param is passed to the numpy.einsum for optimized broadcasted multiplication and reduction sum
    with no optimization the broadcasting with einsum is already faster because it doesn't create copies of intermediate arrays.
    """
    match kernel.shape:
        case (_,):
            windows = _stream2windows(tensor, kernel.shape)
            subscript = "ij,j->i"
        case (_, _):
            windows = _image2windows(tensor, kernel.shape)
            subscript = "ijkl,kl->ij"
        case (_, _, _):
            windows = _space2windows(tensor, kernel.shape)
            subscript = "ijklmn,lmn->ijk"
        case _:
            raise ValueError("This type of convolution needs a custom implementation.")

    # broadcast multiplication and sum over the last kernel.ndim axis
    return einsum(subscript, windows, kernel, optimize=optimize)


def operator_convolution(matrix: Variable, kernel: Variable):
    forward = Variable(convolve_forward(matrix.value, kernel.value))
    inputs = (matrix, kernel)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dConvResult,) = dLoss_dOutputs

        # Gradient w.r.t. input matrix, magically equal to:
        # convolution between padded dLoss_dConvResult and a rotated kernel by 180 degrees
        kernel_flipped = kernel.flip()
        pad_height = kernel.value.shape[0] - 1
        pad_width = kernel.value.shape[1] - 1
        padded_conv_result = dLoss_dConvResult.pad(
            ((pad_height, pad_height), (pad_width, pad_width))
        )
        dLoss_dMatrix = padded_conv_result.convolve(kernel_flipped)

        # Gradient w.r.t. kernel magically equal to:
        # convolution between input matrix and dLoss_dConvResult
        dLoss_dKernel = matrix.convolve(dLoss_dConvResult)

        dLoss_dInputs = (dLoss_dMatrix, dLoss_dKernel)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def activation_ReLU(var: Variable):
    """Element-wise Rectified Linear Unit activation function"""

    forward = Variable(maximum(0, var.value))
    inputs = (var,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dResult,) = dLoss_dOutputs
        dLoss_dInput = dLoss_dResult * Variable((var.value > 0).astype(float))
        dLoss_dInputs = (dLoss_dInput,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def activation_sigmoid(var: Variable):
    """Element-wise Sigmoid activation function"""

    forward = Variable(1.0 / (1.0 + exp(-var.value)))
    inputs = (var,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dResult,) = dLoss_dOutputs
        dLoss_dInput = dLoss_dResult * (forward * (1 - forward))
        dLoss_dInputs = (dLoss_dInput,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def activation_tanh(var: Variable):
    """
    Element-wise Tanh activation function

    tanh(x) = (e^(x) - e^(-x)) / (e^(x) + e^(-x))
    """

    forward = Variable(tanh(var.value))
    inputs = (var,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dResult,) = dLoss_dOutputs
        dLoss_dInput = dLoss_dResult * (1 - forward * forward)
        dLoss_dInputs = (dLoss_dInput,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


def activation_log_softmax(var: Variable, axis: int = -1):
    """
    Element-wise Log(Softmax(var)) activation function along the specified axis

    Use it with Negative-Log-Likelihood to achieve cross-entropy loss.

    Note that for regular softmax the derivatives would have looked like
        dSoftmax_i/dInput_i = Softmax_i * (1 - Softmax_i)
        dSoftmax_i/dInput_j = -Softmax_i * Softmax_j; if i != j
    however with log it simplifies to:
        dLogSoftmax_i/dInput_i = 1 - Softmax_i
        dLogSoftmax_i/dInput_j = -Softmax_j; if i != j

    This implementation includes the standard numerical stability trick with the max subtraction.
    """

    var_value = var.value
    normalized_exp_values = exp(var_value - var_value.max(axis=axis, keepdims=True))
    # log-sum-exp stability trick
    log_softmax_values = normalized_exp_values - log(
        normalized_exp_values.sum(axis=axis, keepdims=True)
    )
    forward = Variable(log_softmax_values)

    inputs = (var,)
    outputs = (forward,)

    def back_fn(dLoss_dOutputs: Sequence[Variable]) -> Sequence[Variable]:
        (dLoss_dResult,) = dLoss_dOutputs
        # Jacobian matrix for log_softmax is complex but gives more control over the learning process
        dLoss_dInput = (
            dLoss_dResult.broadcast_to((*var.shape, *dLoss_dResult.shape)) @ -forward
            + dLoss_dResult
        )
        dLoss_dInputs = (dLoss_dInput,)
        return dLoss_dInputs

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return forward


# TODO debug
@Variable.convert_input_into_variable
def loss_mse(
    predicted: Variable, target: Variable | numeric | Iterable[numeric]
) -> Variable:
    """
    Mean Squared Error Loss between predicted and target variables.
    """
    n_samples = predicted.shape[0]

    err = subtract(predicted.value, target.value)
    mse_value = array(mean(square(err)))
    mse = Variable(mse_value)
    normalization_const = Variable.constant(1.0 / predicted.size)

    inputs = (predicted,)  # the target is a constant, so skip it here
    outputs = (mse,)
    def back_fn(dLoss_dOutputs):
        (dLoss_dOutput,) = dLoss_dOutputs
        # ignore the factor of 2
        dLoss_dInput = dLoss_dOutput * err * normalization_const.broadcast_to(predicted.shape),
        return (dLoss_dInput,)

    global _tape_stack
    _tape_stack.append(Tape(outputs=outputs, inputs=inputs, back_fn=back_fn))

    return mse


@Variable.convert_input_into_variable
def loss_NLL(log_probs: Variable, target_probs: Variable) -> Variable:
    """
    Negative Log-Likelihood Loss for classification tasks.

    :param log_probs: Log probabilities from the model (output of log_softmax).
    :param target_probs: True underlying probability indices (class labels).
    :return: NLL loss variable. Combined with log_softmax gives cross-entropy loss.
    """
    n_samples = log_probs.value.shape[0]
    nll_values = -log_probs.value[range(n_samples), target_indices]
    nll_loss = Variable(nll_values.sum() * (1.0 / n_samples))
    return nll_loss


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
        a dictionary of the d(loss_variable)/d[key] where key is any other variable used to compute the loss_variable

    :params
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
        loss_variable: Variable(ones_like(loss_variable.value))
        if directional_grad is None
        else directional_grad
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

        # due to unconstrained computation graph shape we actually sum all the gradients
        # the automatic topological sorting of the tape ensures we use the final dLoss_d[tape_input] value
        # only when it was fully computed
        for tape_input, dL_dInput in zip(tape_record.inputs, dLoss_dInputs):
            # we could have used defaultdict(lambda x:zeros) but this way we keep the notion of what was used in the process
            if tape_input not in dLoss_d:
                dLoss_d[tape_input] = dL_dInput
            else:
                dLoss_d[tape_input] += dL_dInput

    # debug information values of each intermediate gradient
    # for name, value in dLoss_d.items():
    #     print(f"d{loss_variable.name}_d{name} = {value.name}")

    return dLoss_d
