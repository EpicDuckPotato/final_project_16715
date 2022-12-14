import sys
sys.path.append('src')
import numpy as np
import matplotlib.pyplot as plt
from controller import *
from verification import *
from scipy.linalg import solve_continuous_are, expm
from time_reversed_vanderPol import *

def main(args=None):
  # Test on vanderpol
  xgoal = np.array([0, 0])
  ugoal = np.array([0])

  model = TimeReversedVanderPol()
  nx, nu = model.get_dim()
  # Infinite-horizon LQR
  A, B = model.lin_dynamics(xgoal, ugoal)
  Q = np.eye(nx)
  R = np.eye(nu)
  S = solve_continuous_are(A, B, Q, R)
  K = np.linalg.solve(R, B.transpose()@S)
  policy = LQRPolicy(xgoal, ugoal, S, K)

  print('Van der Pol ROA search')
  # rho = find_roa(model, policy)
  rho1 = find_roa_sample(model, policy)
  # rho2 = find_roa_simulation_2d(model, policy)
  
if __name__ == '__main__':
  main()
