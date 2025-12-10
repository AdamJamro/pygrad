import numpy as np

from network.network import Network, Variable, Linear, clear_tape, loss_mse, _tape_stack


def test_modularity_params():
    """Test modularity of the test structure itself."""

    class FooNN(Network):
        def __init__(self):
            super().__init__()
            self.param = Variable(1.0)

        def forward(self, x: Variable) -> Variable:
            pass

    foo_nn = FooNN()
    assert hasattr(foo_nn, "_parameters")
    assert len(foo_nn._parameters) == 1
    assert foo_nn._parameters[0].value == 1.0


def test_linear_step_in_network():
    """Test integration of a simple linear neural network."""

    class LinearNN(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(2, 2)

        def forward(self, x: Variable) -> Variable:
            return self.L1(x).sum()

    lin_nn = LinearNN()
    input_var = Variable(np.array([[3.0], [4.0]]))
    input_var_2 = Variable(np.array([[1.0], [2.0]]))

    for step in range(50):
        history = []
        for i in range(10):
            out = lin_nn.forward(input_var)
            loss = out.sum()
            history.append(loss.value)
            grad = loss.backward()
            for p in lin_nn.parameters():
                p.value -= 0.02 * grad[p]
            clear_tape()

        print()
        print(history)
        history = []
        for i in range(10):
            out = lin_nn.forward(input_var_2)
            loss = out.sum()
            history.append(loss.value)
            grad = loss.backward()
            for p in lin_nn.parameters():
                p.value += 0.05 * grad[p]
            clear_tape()
        print()
        print(history)

        print(f'epoch {step} minimizing:{lin_nn(input_var)}')
        print(f'epoch {step} maximizing:{lin_nn(input_var_2)}')


    print()
    print(history)
    assert history
    assert set(lin_nn.parameters()) == set(lin_nn._parameters).union(
        set(lin_nn.L1._parameters)
    )  # 2 weights + 1



def test_mlp_network():
    """Test integration of a simple linear neural network."""

    class MLP(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(3, 2)
            self.L2 = Linear(2, 5)
            self.L3 = Linear(5, 2)

        def forward(self, x: Variable) -> Variable:
            x = Variable.ReLU(self.L1(x))
            x = Variable.ReLU(self.L2(x))
            return self.L3(x)


    mlp = MLP()
    print()
    print(list(mlp.parameters()))
    print()
    print(list(mlp.subnetworks()))
    print()
    print(list(mlp.L1.parameters()))

    print()
    print()
    input_var = Variable(np.array([[3.0], [4.0], [5.0]]))
    target_var = Variable(np.array([[10.0], [0.0]]))

    history = []
    for i in range(100):
        out = mlp.forward(input_var)
        loss = loss_mse(out, target_var)
        history.append(loss.value)
        grad = loss.backward()
        for p in mlp.parameters():
            p.value -= 0.05 * grad[p]
        clear_tape()
    print()
    print(history)



def test_mlp_network_mse():
    """Test integration of a simple linear neural network."""

    class MLP(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(3, 2)
            self.L2 = Linear(2, 5)
            self.L3 = Linear(5, 2)

        def forward(self, x: Variable) -> Variable:
            x = Variable.ReLU(self.L1(x))
            x = Variable.ReLU(self.L2(x))
            return self.L3(x)


    mlp = MLP()
    print()
    print(list(mlp.parameters()))
    print()
    print(list(mlp.subnetworks()))
    print()
    print(list(mlp.L1.parameters()))

    print()
    print()
    input_var = Variable(np.array([[3.0], [4.0], [5.0]]))
    target_var = Variable(np.array([[10.0], [0.0]]))

    history = []
    for i in range(100):
        out = mlp.forward(input_var)
        loss = loss_ase(out, target_var)
        history.append(loss.value)
        grad = loss.backward()
        for p in mlp.parameters():
            p.value -= 0.05 * grad[p]
        clear_tape()
    print()
    print(history)

