"""
The backward pass is computed via a topologically sorted stack of callbacks aka the backward pass is "tape-based".
Tape class organizes and singles out Variable's responsibility of storing the computation graph (DAG).
Additionally, it is a neat way we topologically sort the computation graph,
since the computation is going to be performed in the same order we put it on the tape.
"""
from __future__ import annotations
from typing import NamedTuple, Sequence, Callable


# from autodiff.backward_function import BackFunction
class Tape(NamedTuple):
    """
    Single tape record containing information about single computation step in the graph
    reamrk that the back_fn callback operates on Variable's
    that entails that a new graph is being built during the backward pass
    """
    outputs: Sequence[Variable, ...]
    inputs: Sequence[Variable, ...]
    # back_fn: BackFunction
    back_fn: Callable[[Sequence[Variable | None, ...]], Sequence[Variable, ...]]
