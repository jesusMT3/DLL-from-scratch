"""
A function that can train a neural network.
"""
import matplotlib.pyplot as plt
from DLL.tensor import Tensor
from DLL.nn import NeuralNet
from DLL.loss import Loss, MSE
from DLL.optim import Optimizer, SGD
from DLL.data import DataIterator, BatchIterator
from typing import List

def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    data = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)

        data.append(epoch_loss)

    return data

def print_train(data: List[float]) -> None:
    data_size = [i for i in range(1, len(data) + 1)]
    fig, ax = plt.subplots()
    ax.plot(data_size, data)
    plt.show()