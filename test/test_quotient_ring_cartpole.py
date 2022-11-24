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

  print('Pendulum ROA search')
  rho = find_roa(model, policy)
  rho1 = find_roa_sample(model, policy)

if __name__ == '__main__':
  main()
