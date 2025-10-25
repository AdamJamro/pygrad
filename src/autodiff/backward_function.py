from abc import abstractmethod
from typing import Callable

from src.autodiff.variable import Variable

_operator_name_id = 0


def _fresh_backward_fn_name(operator_name: str = None):
    global _operator_name_id
    name = f"__backward_{operator_name}{_operator_name_id}"
    _operator_name_id += 1
    return name


# TODO it is still unused
class BackFunction:
    """
    Keeps the record of how to compute partial derivatives of graph's top node w.r.p. to inputs
    given the partial derivatives of graph's top node w.r.p. to outputs.

    back_fn:
        is the single building block of the backward calculation graph

    """

    def __init__(
        self,
        back_fn: Callable[[tuple[Variable | None, ...]], tuple[Variable, ...]],
        custom_name: str | None,
    ):
        self.back_fn = back_fn
        self.name = custom_name or _fresh_backward_fn_name("custom")

    @abstractmethod
    def __call__(self, arguments: tuple[Variable | None, ...]) -> tuple[Variable, ...]:
        """
        if an argument variable is None it means we didn't reach it before the loss in which respect we do the backward pass
        :param arguments:
        :return: Partial derivatives/jacobians w.r.p. to the arguments

        Note the returned set of Variables is being put back on tape.

        Note new variables will have different names,
        and will expect that the actual value of the variable is not being mutated between calls

        Note the new graph can compute next order derivative with respect to the originally used top node (the Loss)
        """
        pass
