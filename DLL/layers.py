"""
The units that neural networks are made of.
Each layer needs to pass its inputs forward
and propagate gradients backwards.
"""
from typing import Dict, Callable
import numpy as np
from DLL.tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) ->Tensor:
        """
        Produce the outputs corresponding to the inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate the gradient through the layer
        """
        raise NotImplementedError
    
class Linear(Layer):
    """
    Computes: output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        # Inputs will be [batch_size, input_size]
        # Outputs will be [batch_size, input_size]
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)
        
    def forward(self, inputs: Tensor) -> Tensor:
        """
        Outputs: inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    def backward(self, grad: Tensor) ->Tensor:
        """
        if y = f(x) and  = a @ b + c, 
        dy/da = f'(x) @ b.T
        dy/db = a.T @ f'(x)
        dy/dc = f'(x)
        """
        
        self.grads["b"] = np.sum(grad, axis = 0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T
    
    
F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    An activation layer applies a function elementwise 
    to its inputs
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
        
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Chain rule: if y = f(x) and x = g(z), then
        dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad
        
def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)