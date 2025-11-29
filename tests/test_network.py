from typing import Sequence


from network.network import Network, Variable, Linear


def test_modularity_params():
    """Test modularity of the test structure itself."""

    class FooNN(Network):
        def __init__(self):
            super().__init__()
            self.param = Variable(1.0)

    foo_nn = FooNN()
    assert hasattr(foo_nn, "_parameters")
    assert len(foo_nn._parameters) == 1
    assert foo_nn._parameters[0].value == 1.0


def test_linear_step_in_network():
    """Test integration of a simple linear neural network."""

    class LinearNN(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(2, 1)

        def forward(self, x: Variable) -> Sequence[Variable]:
            return self.L1(x)

    lin_nn = LinearNN()
    input_var = Variable([3.0, 4.0])
    output_var = lin_nn.forward(input_var)

    assert output_var.shape == (1,)
    assert set(lin_nn.parameters()) == set(lin_nn._parameters).union(
        set(lin_nn.L1._parameters)
    )  # 2 weights + 1
