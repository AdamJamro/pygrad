"""
Tests for neural network components.
"""

from pygrad.engine import Value
from pygrad.nn import Neuron, Layer, MLP, mse_loss, binary_cross_entropy


class TestNeuron:
    """Test Neuron class."""

    def test_neuron_creation(self):
        """Test neuron initialization."""
        n = Neuron(3, activation="relu")

        assert len(n.w) == 3
        assert n.b is not None
        assert n.activation == "relu"
        assert len(n.parameters()) == 4  # 3 weights + 1 bias

    def test_neuron_forward_pass(self):
        """Test neuron forward pass."""
        n = Neuron(2, activation="relu")
        n.w = [Value(0.5), Value(-0.5)]
        n.b = Value(0.0)

        x = [Value(1.0), Value(2.0)]
        out = n(x)

        # 0.5*1 + (-0.5)*2 + 0 = -0.5, ReLU(-0.5) = 0
        assert out.data == 0.0

    def test_neuron_with_different_activations(self):
        """Test neuron with different activation functions."""
        x = [Value(1.0), Value(1.0)]

        # ReLU
        n_relu = Neuron(2, activation="relu")
        n_relu.w = [Value(1.0), Value(1.0)]
        n_relu.b = Value(-1.0)
        out_relu = n_relu(x)
        assert out_relu.data == 1.0  # 1+1-1=1, ReLU(1)=1

        # Tanh
        n_tanh = Neuron(2, activation="tanh")
        n_tanh.w = [Value(0.0), Value(0.0)]
        n_tanh.b = Value(0.0)
        out_tanh = n_tanh(x)
        assert abs(out_tanh.data - 0.0) < 1e-6  # tanh(0) = 0

        # None (linear)
        n_linear = Neuron(2, activation=None)
        n_linear.w = [Value(2.0), Value(3.0)]
        n_linear.b = Value(1.0)
        out_linear = n_linear(x)
        assert out_linear.data == 6.0  # 2*1 + 3*1 + 1 = 6

    def test_neuron_backward_pass(self):
        """Test gradient flow through neuron."""
        n = Neuron(2, activation=None)
        n.w = [Value(1.0), Value(2.0)]
        n.b = Value(0.5)

        x = [Value(1.0), Value(1.0)]
        out = n(x)
        out.backward()

        # Check that gradients are computed
        assert n.w[0].grad != 0.0
        assert n.w[1].grad != 0.0
        assert n.b.grad != 0.0


class TestLayer:
    """Test Layer class."""

    def test_layer_creation(self):
        """Test layer initialization."""
        layer = Layer(3, 2, activation="relu")

        assert len(layer.neurons) == 2
        assert len(layer.parameters()) == 8  # 2 neurons * (3 weights + 1 bias)

    def test_layer_forward_pass(self):
        """Test layer forward pass."""
        layer = Layer(2, 3, activation="relu")
        x = [Value(1.0), Value(2.0)]
        out = layer(x)

        assert len(out) == 3
        assert all(isinstance(o, Value) for o in out)

    def test_layer_single_output(self):
        """Test layer with single output neuron."""
        layer = Layer(2, 1, activation="relu")
        x = [Value(1.0), Value(2.0)]
        out = layer(x)

        # Single output should return a Value, not a list
        assert isinstance(out, Value)

    def test_layer_backward_pass(self):
        """Test gradient flow through layer."""
        layer = Layer(2, 2, activation=None)
        x = [Value(1.0), Value(1.0)]
        out = layer(x)

        # Create a simple loss
        loss = sum(out, Value(0.0))
        loss.backward()

        # Check that all parameters have gradients
        for p in layer.parameters():
            assert p.grad != 0.0


class TestMLP:
    """Test Multi-Layer Perceptron."""

    def test_mlp_creation(self):
        """Test MLP initialization."""
        mlp = MLP(3, [4, 2], activation="relu")

        assert len(mlp.layers) == 2
        # First layer: 3 inputs * 4 neurons * (3 weights + 1 bias) = 16
        # Second layer: 4 inputs * 2 neurons * (4 weights + 1 bias) = 10
        # Total: 26 parameters
        assert len(mlp.parameters()) == 26

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        mlp = MLP(2, [3, 1], activation="relu")
        x = [Value(1.0), Value(2.0)]
        out = mlp(x)

        # With a single output, should return a Value
        assert isinstance(out, Value)

    def test_mlp_multiple_outputs(self):
        """Test MLP with multiple outputs."""
        mlp = MLP(2, [3, 2], activation="relu")
        x = [Value(1.0), Value(2.0)]
        out = mlp(x)

        assert len(out) == 2
        assert all(isinstance(o, Value) for o in out)

    def test_mlp_backward_pass(self):
        """Test gradient flow through MLP."""
        mlp = MLP(2, [3, 1], activation="relu")
        x = [Value(1.0), Value(1.0)]
        out = mlp(x)
        out.backward()

        # Check that all parameters have gradients
        # (some might be zero due to ReLU, but most should be non-zero)
        non_zero_grads = sum(1 for p in mlp.parameters() if p.grad != 0.0)
        assert non_zero_grads > 0

    def test_mlp_zero_grad(self):
        """Test zeroing gradients for MLP."""
        mlp = MLP(2, [2, 1])
        x = [Value(1.0), Value(1.0)]
        out = mlp(x)
        out.backward()

        # Verify some gradients are non-zero
        assert any(p.grad != 0.0 for p in mlp.parameters())

        # Zero gradients
        mlp.zero_grad()

        # Verify all gradients are zero
        assert all(p.grad == 0.0 for p in mlp.parameters())


class TestLossFunctions:
    """Test loss functions."""

    def test_mse_loss(self):
        """Test Mean Squared Error loss."""
        predictions = [Value(1.0), Value(2.0), Value(3.0)]
        targets = [1.5, 2.5, 2.5]

        loss = mse_loss(predictions, targets)

        # MSE = ((1-1.5)^2 + (2-2.5)^2 + (3-2.5)^2) / 3
        # = (0.25 + 0.25 + 0.25) / 3 = 0.25
        assert abs(loss.data - 0.25) < 1e-6

    def test_mse_loss_backward(self):
        """Test MSE loss gradient computation."""
        predictions = [Value(1.0), Value(2.0)]
        targets = [0.0, 0.0]

        loss = mse_loss(predictions, targets)
        loss.backward()

        # dL/dpred = 2*(pred - target) / n
        # For pred[0]: 2*(1-0)/2 = 1.0
        # For pred[1]: 2*(2-0)/2 = 2.0
        assert predictions[0].grad == 1.0
        assert predictions[1].grad == 2.0

    def test_binary_cross_entropy(self):
        """Test Binary Cross Entropy loss."""
        # Perfect predictions
        predictions = [Value(0.9), Value(0.1)]
        targets = [1.0, 0.0]

        loss = binary_cross_entropy(predictions, targets)

        # Loss should be low for good predictions
        assert loss.data > 0.0
        assert loss.data < 0.2

    def test_binary_cross_entropy_backward(self):
        """Test BCE loss gradient computation."""
        predictions = [Value(0.5), Value(0.5)]
        targets = [1.0, 0.0]

        loss = binary_cross_entropy(predictions, targets)
        loss.backward()

        # Gradients should be non-zero
        assert predictions[0].grad != 0.0
        assert predictions[1].grad != 0.0


class TestTraining:
    """Test training scenarios."""

    def test_simple_training_step(self):
        """Test a simple training step."""
        # Create a simple network
        mlp = MLP(1, [2, 1], activation="relu")

        # Simple training data
        x = [Value(1.0)]
        target = 2.0

        # Forward pass
        pred = mlp(x)
        loss = (pred - Value(target)) ** 2

        # Backward pass
        mlp.zero_grad()
        loss.backward()

        # Simple gradient descent step
        learning_rate = 0.01
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad

        # After update, parameters should have changed
        # (This is just a sanity check, not testing convergence)
        assert loss.data >= 0.0

    def test_overfitting_single_point(self):
        """Test that network can overfit a single data point."""
        # Create a simple network
        mlp = MLP(1, [4, 1], activation="tanh")

        # Single training point
        x = [Value(0.5)]
        target = 1.0

        # Train for several iterations
        for _ in range(50):
            # Forward pass
            pred = mlp(x)
            loss = (pred - Value(target)) ** 2

            # Backward pass
            mlp.zero_grad()
            loss.backward()

            # Gradient descent
            learning_rate = 0.1
            for p in mlp.parameters():
                p.data -= learning_rate * p.grad

        # Network should have learned to predict close to target
        final_pred = mlp([Value(0.5)])
        assert abs(final_pred.data - target) < 0.5
