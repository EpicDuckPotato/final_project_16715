import numpy as np
from scipy.linalg import solve_continuous_are, expm
from pydrake.symbolic import *

class Pendulum(object):
  def __init__(self):
    self.nq = 2
    self.nv = 1
    self.nq_minimal = self.nv
    self.nx = self.nq + self.nv
    self.nx_minimal = self.nq_minimal + self.nv
    self.nu = 1

  def get_dim(self):
    return self.nx_minimal, self.nu

  def dynamics(self, state, u):
    theta = state[0]
    thetadot = state[1]
    s = np.sin(theta)
    thetaddot = u[0] - thetadot + s
    return np.array([thetadot, thetaddot])

  def lin_dynamics(self, state, u):
    theta = state[0]

    c = np.cos(theta)

    A = np.zeros((2, 2))
    B = np.zeros((2, 1))

    A[0, 1] = 1
    A[1, 0] = c
    A[1, 1] = -1

    B[1, 0] = 1

    return A, B

  def generate_drake_variables(self):
    c_rev = MakeVectorContinuousVariable(self.nv, 'c_rev')
    s_rev = MakeVectorContinuousVariable(self.nv, 's_rev')
    v = MakeVectorContinuousVariable(self.nv, 'v')
    vdot = MakeVectorContinuousVariable(self.nv, 'vdot')
    u = MakeVectorContinuousVariable(self.nu, 'u')
    return np.concatenate((c_rev, s_rev)), v, vdot, u

  # transform from [qdot; vdot] = T*[v; vdot], where q = [c0, s0, ...]
  def get_T(self, q):
    T = np.zeros((self.nq + self.nv, self.nx_minimal), dtype=Expression)
    T[0, 0] = -q[1] # Deriv of cos = -sin
    T[1, 0] = q[0] # Deriv of sin = cos
    T[-self.nv:, -self.nv:] = np.eye(self.nv, dtype=Expression)
    return T

  # Converts from sympy to Drake
  def get_drake_constraints(self, q, v, vdot, u):
    # Acceleration constraint, then trig constraint
    constraints = np.zeros(2, dtype=Expression)

    # Acceleration constraint
    constraints[0] = u[0] - v[0] - q[1]

    # Trig constraint, s^2 + c^2 - 1 = 0
    constraints[1] = q[0]**2 + q[1]**2 - 1

    return constraints

  def trig_lqr(self):
    # LQR in minimal coordinates
    A, B = self.lin_dynamics(np.zeros(self.nx_minimal), np.zeros(self.nu))
    Q = np.eye(self.nx_minimal)
    R = np.eye(self.nu)
    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.transpose()@S)

    # Convert to trigonometric coordinates
    Ktrig = np.zeros((self.nu, self.nq + self.nv))
    Ktrig[:, -self.nv:] = K[:, -self.nv:]
    Strig = np.zeros((self.nq + self.nv, self.nq + self.nv))
    Strig[-self.nv:, -self.nv:] = S[-self.nv:, -self.nv:]

    # Columns corresponding to theta in minimal coordinates correspond to sin(theta) in trig coordinates
    Ktrig[:, 1] = K[:, 0]

    # (sin, sin)
    Strig[1, 1] = S[0, 0]

    # (sin, v)
    Strig[1, -self.nv:] = S[0, -self.nv:]

    # (v, sin)
    Strig[-self.nv:, 1] = S[-self.nv:, 0]

    return Strig, Ktrig, S, K
