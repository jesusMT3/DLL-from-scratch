"""
A loss function measures how good our predictions are.
We can use this to adjust the parameters of our neural network.
"""
import numpy as np
from DLL.tensor import Tensor

class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    Mean Square Error loss function, although it's going to be total 
    squared error
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)