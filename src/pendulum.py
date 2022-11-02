'''DYNAMICS MODEL OF PENDULUM
All parameters are unit values
'''

import numpy as np
from pydrake.symbolic import cos, sin
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian

class Pendulum(object):
  def __init__(self):
    self.m = 1
    self.n = 2
    # xg = np.array(xg)
    # ug = np.array([ug])

  #   if xg.shape(0) != self.n:
  #     print(f"Goal state should be an array with dim of {xg.shape(0)}")
  #     exit
  #   else:
  #     self.xg = np.copy(xg)

  #   if ug.shape(0) != self.m:
  #     print(f"Goal control dimension should with dim of {ug.shape(0)}")
  #     exit
  #   else:
  #     self.ug = np.copy(ug)  

  # def set_goal(self, xg, ug):
  #   if xg.shape(0) != self.n:
  #     print(f"Goal state should be an array with dim of {xg.shape(0)}")
  #     exit
  #   else:
  #     self.xg = np.copy(xg)

  #   if ug.shape(0) != self.m:
  #     print(f"Goal control dimension should with dim of {ug.shape(0)}")
  #     exit
  #   else:
  #     self.ug = np.copy(ug)  

  def get_dim(self):
    return self.n, self.m

  def dynamics(self, state, u):
    theta = state[0]
    thetadot = state[1]
    s = np.sin(theta)
    thetaddot = u[0] - thetadot - s
    return np.array([thetadot, thetaddot])

  def lin_dynamics(self, state, u):
    theta = state[0]

    c = np.cos(theta)

    A = np.zeros((2, 2))
    B = np.zeros((2, 1))

    A[0, 1] = 1
    A[1, 0] = -c
    A[1, 1] = -1

    B[1, 0] = 1

    return A, B

  def sym_dynamics(self, state, policy, t=0):
    th = state[0]
    thdot = state[1]
    u = policy.get_u(state, t)
    s = sin(th)
    thddot = u[0] - thdot - s
    return np.array([thdot, thddot])