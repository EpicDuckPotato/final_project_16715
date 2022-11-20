'''DYNAMICS MODEL OF VAL DER POL OSCILLATOR
https://openmdao.github.io/dymos/examples/vanderpol/vanderpol.html
'''

import numpy as np
from pydrake.symbolic import cos, sin
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian

class TimeReversedVanderPol(object):
  def __init__(self, mu=1):
    self.m = 1
    self.n = 2
    self.mu = mu

  # Known Lyapunov function for the time-reversed oscillator (corresponding to mu = -1)
  def known_V(self, state):
    x1 = state[0]
    x2 = state[1]
    V = (1.8027e-06) + (0.28557) * x1**2 + (0.0085754) * x1**4 + \
        (0.18442) * x2**2 + (0.016538) * x2**4 + \
        (-0.34562) * x2 * x1 + (0.064721) * x2 * x1**3 + \
        (0.10556) * x2**2 * x1**2 + (-0.060367) * x2**3 * x1
    return V

  def get_dim(self):
    return self.n, self.m

  def dynamics(self, state, u):
    x0 = state[0]
    x1 = state[1]
    dx0 = -x1
    dx1 = -(1 - x0**2)*x1 + x0
    return np.array([dx0, dx1])

  def sym_dynamics(self, state, policy, t=0):
    u = policy.get_u(state, t)
    x0 = state[0]
    x1 = state[1]
    dx0 = -x1
    dx1 = -(1 - x0**2)*x1 + x0
    return np.array([dx0, dx1])
