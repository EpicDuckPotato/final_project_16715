'''DYNAMICS MODEL OF VAL DER POL OSCILLATOR
https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html
'''

import numpy as np
from pydrake.symbolic import cos, sin
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian

class VanderPol(object):
  def __init__(self):
    self.m = 1
    self.n = 2

  def get_dim(self):
    return self.n, self.m

  def dynamics(self, state, u):
    x0 = state[0]
    x1 = state[1]
    dx0 = (1 - x1**2)*x0 - x1 + u[0]
    dx1 = x0
    return np.array([dx0, dx1])

  def lin_dynamics(self, state, u):
    x0 = state[0]
    x1 = state[1]

    A = np.zeros((2, 2))
    B = np.zeros((2, 1))

    A[0, 0] = 1 - x1**2
    A[0, 1] = -1 - 2*x0*x1
    A[1, 0] = 1
    A[1, 1] = 0

    B[0, 0] = 1

    return A, B

  def sym_dynamics(self, state, policy, t=0):
    u = policy.get_u(state, t)
    x0 = state[0]
    x1 = state[1]
    dx0 = (1 - x1**2)*x0 - x1 + u[0]
    dx1 = x0
    return np.array([dx0, dx1])