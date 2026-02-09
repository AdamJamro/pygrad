# PyGrad

A lightweight autodiff engine with implementing backpropagation (reverse-mode autodiff) over a dynamically built DAG. Equipped in a simple neural network interface on top (PyTorch-like API).

All operators are optimized for vectorized calculations (ndarray nodes keep track of jacobians instead of scalars). Some operators are capable of handling tensors. This project's purpose is to aid prototyping or learning about the backprop algorithm.
The implementation relies only on numpy.

Automatic differentiation is based on tape-stack that auto-magically handles topological sorting of the DAG.

## Features

- **Autograd Engine**: Automatic differentiation through computational graph tracking
- **Neural Network Building Blocks**: Multi-Layer Perceptrons (MLPs), Convolutional Layers 
- **Pure Python**: implementation based on numpy's interface for fortran/C-speed mp.arrays multiplication
- **Easy to extend**: the backprop algo is fixed, simply stash more operators to extend the functionality. 
New operators need to adhere engine's api, there are many examples of already implemented operators. 


## Installation

```bash
# Clone the repository
git clone https://github.com/AdamJamro/pygrad.git
cd pygrad

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Autograd Example

```python
from autodiff import Variable

# Create computational graph
x = Variable(3.0)
y = Variable(2.0)
z = (x + y) * (x - y) # any function using predefined operators

# Compute gradients
dz_d = z.backward()

print(f"∂z/∂x = {dz_d[z]}")  # ∂z/∂z = 1.0
print(f"∂z/∂x = {dz_d[x]}")  # ∂z/∂x = 6.0
print(f"∂z/∂y = {dz_d[y]}")  # ∂z/∂y = -4.0
```

### Neural Network Example

```python
from network import Network, Variable, Linear, mse_loss, ReLU

# Create a Multi-Layer Perceptron
# Architecture: 2 inputs -> 4 hidden neurons -> 1 output
class MLP(Network):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self._subnetworks.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            self._subnetworks.append(ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
mlp = MLP(2, [4, 1])

# Training data
X = [[Variable(0.0), Variable(1.0)], [Variable(1.0), Variable(0.0)]]
y = [Variable(1.0), Variable(1.0)]

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = [mlp(x) for x in X]
    loss = mse_loss(predictions, y)
    
    # Backward pass
    mlp.zero_grad()
    loss.backward()
    
    # Gradient descent
    learning_rate = 0.01
    for param in mlp.parameters():
        param.data -= learning_rate * param.grad
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

## Core Components

### Value Class

The `Value` class is the core of the autograd engine. It wraps scalar values and tracks operations to build a computational graph.

**Supported Operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Activation functions: `relu()`, `tanh()`, `sigmoid()`, ...

### Neural Network Components

#### Neuron
A single neuron with weights, bias, and activation function.

```python
from pygrad_obsolete.nn import Neuron

neuron = Neuron(nin=3, activation='relu')
output = neuron([Value(1.0), Value(2.0), Value(3.0)])
```

#### Layer
A layer of multiple neurons.

```python
from pygrad_obsolete.nn import Layer

layer = Layer(nin=3, nout=4, activation='relu')
outputs = layer([Value(1.0), Value(2.0), Value(3.0)])
```

#### MLP (Multi-Layer Perceptron)
A complete neural network with multiple layers.

```python
from pygrad_obsolete import MLP

# Create network: 2 inputs -> 8 hidden -> 4 hidden -> 1 output
mlp = MLP(2, [8, 4, 1], activation='tanh')
```

### Loss Functions

```python
from pygrad_obsolete.nn import mse_loss, binary_cross_entropy

# Mean Squared Error
loss = mse_loss(predictions, targets)

# Binary Cross Entropy
loss = binary_cross_entropy(predictions, targets)
```

## Examples

Run the example script to see PyGrad in action:

```bash
python examples/example.py
```

This demonstrates:
- Basic autograd operations
- Activation functions
- Neural network creation
- Training a network to approximate a function
- Solving the XOR problem

## How It Works

PyGrad implements reverse-mode automatic differentiation (backpropagation):

1. **Forward Pass**: Operations create a computational graph
2. **Backward Pass**: Gradients flow backward through the graph using the chain rule
3. **Topological Sort**: Ensures gradients are computed in the correct order

### Example: Understanding the Computational Graph

```python
from pygrad_obsolete import Value

a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b
c.label = 'c'
d = c + a
d.label = 'd'

d.backward()

# The computational graph:
# a (2.0) ──┬──> * ──> c (6.0) ──┬──> + ──> d (8.0)
#           │                     │
# b (3.0) ──┘                     │
#                                 │
# a (2.0) ────────────────────────┘

print(f"a.grad = {a.grad}")  # 4.0 (gradient accumulates from two paths)
print(f"b.grad = {b.grad}")  # 2.0
print(f"c.grad = {c.grad}")  # 1.0
```

## Architecture

```
pygrad/
├── __init__.py       # Package initialization
├── engine.py         # Autograd engine (Value class)
└── variable.py            # Neural network components (Neuron, Layer, MLP)

tests/
examples/
```

## License

MIT License - See LICENSE file for details
