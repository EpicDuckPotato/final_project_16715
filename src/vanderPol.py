'''DYNAMICS MODEL OF VAL DER POL OSCILLATOR
https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html
'''

import numpy as np
from pydrake.symbolic import cos, sin
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian

class VanderPol(object):
  def __init__(self, mu=1):
    self.m = 1
    self.n = 2
    self.mu = mu

  # Known Lyapunov function for the time-reversed oscillator (corresponding to mu = -1)
  def known_V(self, state):
    x0 = state[0]
    x1 = state[1]
    V = (1.8027e-06) + (0.28557) * x1**2 + (0.0085754) * x1**4 + \
        (0.18442) * x0**2 + (0.016538) * x0**4 + \
        (-0.34562) * x0 * x1 + (0.064721) * x0 * x1**3 + \
        (0.10556) * x0**2 * x1**2 + (-0.060367) * x0**3 * x1
    return V

  def known_Vgrad(self, state):
    x0 = state[0]
    x1 = state[1]
    Vgrad = np.array([2*(0.18442) * x0 + 4*(0.016538) * x0**3 + (-0.34562) * x1 + (0.064721) * x1**3 + 2*(0.10556) * x0 * x1**2 + 3*(-0.060367) * x0**2 * x1, \
                     2*(0.28557) * x1 + 4*(0.0085754) * x1**3 + (-0.34562) * x0 + 3*(0.064721) * x0 * x1**2 + 2*(0.10556) * x0**2 * x1 + (-0.060367) * x0**3])
    return Vgrad

  def get_dim(self):
    return self.n, self.m

  def dynamics(self, state, u):
    x0 = state[0]
    x1 = state[1]
    dx0 = self.mu*(1 - x1**2)*x0 - x1 + u[0]
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
