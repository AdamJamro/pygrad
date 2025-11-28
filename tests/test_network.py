from numpy import ones

from network.network import Network, Variable

def test_modularity_params():
    """Test modularity of the test structure itself."""
    class FooNN(Network):
        def __init__(self):
            super().__init__()
            self.param = Variable(1.0)

    foo_nn = FooNN()
    assert hasattr(foo_nn, '_parameters')
    assert len(foo_nn._parameters) == 1
    assert foo_nn._parameters[0].value == 1.0


def test_integration_LNN():
    """Test integration of a simple linear neural network."""
    class LinearNN(Network):
        def __init__(self):
            super().__init__()
            self.weight = Variable(ones((2,2)))
            self.bias = Variable(ones(2))

        def forward(self, x: Variable) -> Variable:
            return self.weight * x + self.bias

    lin_nn = LinearNN()
    input_var = Variable(3.0)
    output_var = lin_nn.forward(input_var)

    assert output_var.value == 7.0  # = 2*3 + 1