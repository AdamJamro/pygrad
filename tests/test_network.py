import numpy as np

from autodiff.variable import loss_NLL
from network.network import Network, Variable, Linear, clear_tape, loss_mse, loss_mae, _tape_stack, Conv, Conv2d


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




def test_network_internal_tooling():

    class Net(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(1, 2)
            self.L2 = Linear(2, 3)
            self.L3 = Linear(3, 4)
            self.L4 = Conv(1)
            self.L5 = Conv2d(3)
            self.param1 = Variable(2.0)

        def forward(self, x: Variable) -> Variable:
            pass


    mlp = Net()
    print()
    print(list(mlp.parameters()))
    print()
    print(list(mlp.subnetworks()))
    print()
    print(list(mlp.L1.parameters()))
    for subnet in mlp.subnetworks():
        print(f'Subnet: {subnet} with params: {list(subnet.parameters())}')


def test_mlp_network():

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



def test_mlp_network_mae():

    class MLP2(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(3, 2)
            self.L2 = Linear(2, 5)
            self.L3 = Linear(5, 2)

        def forward(self, x: Variable) -> Variable:
            x = Variable.ReLU(self.L1(x))
            x = Variable.ReLU(self.L2(x))
            return self.L3(x)


    mlp = MLP2()

    input_var = Variable(np.array([[3.0], [4.0], [5.0]]))
    target_var = Variable(np.array([[10.0], [0.0]]))

    history = []
    for i in range(100):
        out = mlp.forward(input_var)
        loss = loss_mae(out, target_var)
        history.append(loss.value)
        grad = loss.backward()
        for p in mlp.parameters():
            p.value -= 0.005 * grad[p]
        clear_tape()
    print()
    print(history)




def test_mlp_with_sigmoid():
    class MLP3(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(3, 3)
            self.L2 = Linear(3, 4)
            self.L3 = Linear(4, 2)

        def forward(self, x: Variable) -> Variable:
            x = Variable.sigmoid(self.L1(x))
            x = Variable.sigmoid(self.L2(x))
            return self.L3(x)


    mlp = MLP3()
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



def test_mlp_with_tanh():

    class MLP4(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(3, 5)
            self.L2 = Linear(5, 6)
            self.L3 = Linear(6, 2)

        def forward(self, x: Variable) -> Variable:
            x = Variable.tanh(self.L1(x))
            x = Variable.tanh(self.L2(x))
            return self.L3(x)


    mlp = MLP4()
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



def test_class_predictor():

    class Predictor(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(3, 5)
            self.L2 = Linear(5, 6)
            self.L3 = Linear(6, 2)

        def forward(self, x: Variable) -> Variable:
            x = Variable.ReLU(self.L1(x))
            x = Variable.ReLU(self.L2(x))
            return Variable.log_softmax(self.L3(x), axis=0)


    pred = Predictor()
    input_var = Variable(np.array([[3.0], [4.0], [5.0]]))
    target_var = Variable(np.array([[1.0], [0.0]]))

    history = []
    for i in range(100):
        out = pred.forward(input_var)
        loss = loss_mse(out, target_var)
        history.append(loss.value)
        grad = loss.backward()
        for p in pred.parameters():
            p.value -= 0.01 * grad[p]
        clear_tape()
    print()
    print(history)


def test_nll_loss():

    class Predictor2(Network):
        def __init__(self):
            super().__init__()
            self.L1 = Linear(3, 5)
            self.L2 = Linear(5, 6)
            self.L3 = Linear(6, 4)

        def forward(self, x: Variable) -> Variable:
            x = Variable.ReLU(self.L1(x))
            x = Variable.ReLU(self.L2(x))
            return Variable.log_softmax(self.L3(x), axis=0)


    pred2 = Predictor2()
    input_var = Variable(np.array([[3.0], [4.0], [5.0]]))
    target_var = Variable(np.array([[1.0], [0.0], [0.0], [0.0]]))

    history = []
    for i in range(100):
        out = pred2.forward(input_var)
        loss = loss_NLL(out, target_var)
        history.append(loss.value)
        grad = loss.backward()
        for p in pred2.parameters():
            p.value -= 0.001 * grad[p]
        clear_tape()
    print()
    print(history)


def test_convolutional_network():
    class ConvNet(Network):
        def __init__(self):
            super().__init__()
            self.conv = Conv(8)
            self.L1 = Linear(16, 10)

        def forward(self, x: Variable) -> Variable:
            return self.L1(Variable.ReLU(self.conv(x).reshape_vec_as_mat()))


    conv_net = ConvNet()
    input_var = Variable(np.random.randn(16 + 8 - 1))
    target_var = Variable(np.random.randn(10,1))

    history = []
    for i in range(50):
        out = conv_net.forward(input_var)
        loss = loss_mse(out, target_var)
        history.append(loss.value)
        grad = loss.backward()
        for p in conv_net.parameters():
            p.value -= 0.01 * grad[p]
        clear_tape()
    print()
    print(history)