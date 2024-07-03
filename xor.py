"""
The canonical example of a function that cannot 
be learned with a simple linear model is XOR.
"""
import numpy as np
from DLL.train import train
from DLL.nn import NeuralNet
from DLL.layers import Linear, Tanh 

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size = 2, output_size = 2),
    Tanh(),
    Linear(input_size = 2,  output_size = 2),
])

train(net, inputs, targets, num_epochs = 10000)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)