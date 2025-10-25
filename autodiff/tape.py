"""
The backward pass is lazily computed via callbacks aka the backward pass is "tape-based".
Tape class organizes and singles out Variable's responsibility of storing the computation graph (DAG).
Additionally, it is a neat way we topologically sort the computation graph,
since the computation is going to be performed in the same order we put it on the tape.
"""

from typing import NamedTuple, Callable
