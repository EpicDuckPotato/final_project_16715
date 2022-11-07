import sys
sys.path.append('src')
from sampling import *
import numpy as np
import matplotlib.pyplot as plt
from n_link_cartpole import *
from controller import *
from verification import *
from scipy.linalg import solve_continuous_are, expm

def main(args=None):
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
    return 2*x@S@model.dynamics(x, policy.get_u(x))

  def grad_fn(x):
    A, B = model.lin_dynamics(xgoal, ugoal)
    xdot = model.dynamics(x, policy.get_u(x))
    grad = 2*(xdot@S + x@S@(A - B@K))
    return grad

  samples = []
  num_samples = 100
  for i in range(100):
    if len(samples) >= num_samples:
      break
    samples.extend(sample_isocontours(Vdot_fn, grad_fn, nx, num_samples, alpha=0.1))
  samples = np.array(samples)
  plt.scatter(samples[:, 0], samples[:, 1])
  plt.show()

  w = MakeVectorContinuousVariable(nx, 'w')
  V = np.array([x@S@x for x in samples])
  degV = 2
  rho = check_sos_sample_no_sym(V, w, samples, degV)
  print(rho)

if __name__ == '__main__':
  main()
