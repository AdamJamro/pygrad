"""
Example usage of PyGrad - Neural Network with Autograd.

This example demonstrates:
1. Creating a simple Multi-Layer Perceptron
2. Training it on a simple dataset
3. Using the autograd engine for gradient computation
"""

from pygrad import Value, MLP
from pygrad.nn import mse_loss


def simple_autograd_example():
    """Demonstrate basic autograd functionality."""
    print("=" * 50)
    print("Simple Autograd Example")
    print("=" * 50)
    
    # Create computational graph: f(x, y) = (x + y) * (x - y)
    x = Value(3.0, label='x')
    y = Value(2.0, label='y')
    
    z1 = x + y
    z1.label = 'x+y'
    
    z2 = x - y
    z2.label = 'x-y'
    
    result = z1 * z2
    result.label = 'result'
    
    # Compute gradients
    result.backward()
    
    print(f"x = {x.data}, y = {y.data}")
    print(f"f(x, y) = (x + y) * (x - y) = {result.data}")
    print(f"∂f/∂x = {x.grad}")
    print(f"∂f/∂y = {y.grad}")
    print()


def activation_functions_example():
    """Demonstrate activation functions."""
    print("=" * 50)
    print("Activation Functions Example")
    print("=" * 50)
    
    x = Value(0.5)
    
    print(f"Input: {x.data}")
    print(f"ReLU: {x.relu().data}")
    print(f"Tanh: {x.tanh().data}")
    print(f"Sigmoid: {x.sigmoid().data}")
    print()


def simple_neuron_example():
    """Demonstrate a simple neural network."""
    print("=" * 50)
    print("Simple Neural Network Example")
    print("=" * 50)
    
    # Create a small MLP: 2 inputs -> 4 hidden -> 1 output
    mlp = MLP(2, [4, 1], activation='relu')
    
    print(f"Network architecture: {mlp}")
    print(f"Total parameters: {len(mlp.parameters())}")
    print()
    
    # Single forward pass
    x = [Value(1.0), Value(2.0)]
    output = mlp(x)
    print(f"Input: {[v.data for v in x]}")
    print(f"Output: {output.data}")
    print()


def training_example():
    """Demonstrate training a neural network."""
    print("=" * 50)
    print("Training Example - Simple Function Approximation")
    print("=" * 50)
    
    # Create a small MLP: 1 input -> 8 hidden -> 1 output
    mlp = MLP(1, [8, 1], activation='tanh')
    
    # Training data: approximate y = 2*x - 1
    X = [
        [Value(0.0)],
        [Value(0.5)],
        [Value(1.0)],
        [Value(1.5)],
        [Value(2.0)],
    ]
    y = [
        Value(-1.0),
        Value(0.0),
        Value(1.0),
        Value(2.0),
        Value(3.0),
    ]
    
    # Training loop
    learning_rate = 0.01
    epochs = 100
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(epochs):
        # Forward pass
        predictions = [mlp(x) for x in X]
        loss = mse_loss(predictions, y)
        
        # Backward pass
        mlp.zero_grad()
        loss.backward()
        
        # Gradient descent
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.data:.6f}")
    
    print("\nFinal predictions vs targets:")
    predictions = [mlp(x) for x in X]
    for i, (pred, target) in enumerate(zip(predictions, y)):
        print(f"  Input: {X[i][0].data:.1f} -> Pred: {pred.data:.3f}, Target: {target.data:.3f}")
    print()


def binary_classification_example():
    """Demonstrate binary classification."""
    print("=" * 50)
    print("Binary Classification Example - XOR Problem")
    print("=" * 50)
    
    # Create MLP for XOR: 2 inputs -> 4 hidden -> 1 output
    mlp = MLP(2, [4, 1], activation='tanh')
    
    # XOR training data
    X = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    y = [
        Value(0.0),
        Value(1.0),
        Value(1.0),
        Value(0.0),
    ]
    
    # Training
    learning_rate = 0.1
    epochs = 200
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(epochs):
        # Forward pass
        predictions = [mlp(x) for x in X]
        loss = mse_loss(predictions, y)
        
        # Backward pass
        mlp.zero_grad()
        loss.backward()
        
        # Gradient descent
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.data:.6f}")
    
    print("\nFinal predictions vs targets (XOR):")
    predictions = [mlp(x) for x in X]
    for i, (pred, target) in enumerate(zip(predictions, y)):
        x_vals = [v.data for v in X[i]]
        print(f"  Input: {x_vals} -> Pred: {pred.data:.3f}, Target: {target.data:.1f}")
    print()


if __name__ == "__main__":
    simple_autograd_example()
    activation_functions_example()
    simple_neuron_example()
    training_example()
    binary_classification_example()
    
    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)
