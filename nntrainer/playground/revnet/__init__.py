'''
RevNet tries to solve x given y=f(x) where f(.) is a neural network.
The method is a two-step optimization.
The first step finds a solution x0 s.t. y=f(x0) with gradient descent.
The second step finds x such that f(x)=f(x0) by changing from x0 in directions vertical to the grandient to minimize y changes.
'''
from .revnet import RevNet