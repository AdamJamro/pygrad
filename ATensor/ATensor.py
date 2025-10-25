from numpy import array
from numpy import ndarray


class ATensor:
    """Initially:
    Tensor wrapper that enables core autograd functionality.

    Attributes:
        data (ndarray): The underlying data of the tensor.
    """

    data: ndarray
    _is_parameter: bool = False
    requires_autograd: bool = False

    def __init__(
        self, data, is_parameter: bool = False, requires_autograd: bool = False
    ):
        try:
            if isinstance(data, list) or isinstance(data, tuple):
                data = array(data)
            elif isinstance(data, ndarray):
                data = data.copy()
            elif not isinstance(data, (int, float)):
                raise TypeError("Data must be a list, tuple, ndarray, int, or float.")
            self.data: ndarray = array(data)
        except Exception as e:
            print(f"Error initializing ATensor: {e}")
            raise ValueError(f"Invalid data: {type(data)} type for ATensor.")

        self._is_parameter: bool = is_parameter

    def __repr__(self):
        return f"ATensor({self.data})"

    def __add__(self, other):
        if isinstance(other, ATensor):
            return ATensor(self.data + other.data)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ATensor):
            return ATensor(self.data - other.data)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, ATensor):
            return ATensor(self.data * other.data)
        return NotImplemented
