import sys
sys.path.append('src')
from sampling import *
import numpy as np
import matplotlib.pyplot as plt
from n_link_cartpole import *
from controller import *
from scipy.linalg import solve_continuous_are, expm

def main(args=None):
  # Test on a circle
  f = lambda x: np.array([x[0]**2 + x[1]**2 - 25])
  J_fn = lambda x: np.array([[2*x[0], 2*x[1]]])
  x0 = np.array([0, 5], dtype=np.float64)
  samples = sample_isocontours(f, J_fn, x0, 100)
  plt.plot(samples[:, 0], samples[:, 1])
  plt.show()

  # Test on n-link cartpole
  N = 2 # Number of revolute joints. There is always one prismatic joint, controlling the cart
  link_length = 1
  link_mass = 1

  model = NLinkCartpole(N, link_length, link_mass)
  nx, nu = model.get_dim()
  # Infinite-horizon LQR
  xgoal = np.zeros(nx)
  ugoal = np.zeros(nu)
  A, B = model.lin_dynamics(xgoal, ugoal)
  Q = np.eye(nx)
  R = np.eye(nu)
  S = solve_continuous_are(A, B, Q, R)
  K = np.linalg.solve(R, B.transpose()@S)
  policy = LQRPolicy(xgoal, ugoal, S, K)

  # V = x@S@x
  # Vdot = 2*x@S@xdot
  def Vdot_fn(x):
    return np.array([2*x@S@model.dynamics(x, policy.get_u(x))])

  def J_fn(x):
    J = np.zeros((1, nx))
    A, B = model.lin_dynamics(xgoal, ugoal)
    xdot = model.dynamics(x, policy.get_u(x))
    J[0] = 2*(xdot@S + x@S@(A - B@K))
    return J

  x0 = np.copy(xgoal)
  samples = sample_isocontours(Vdot_fn, J_fn, x0, 1000, alpha=0.1)
  plt.plot(samples[:, 0], samples[:, 1])
  plt.show()

if __name__ == '__main__':
  main()
