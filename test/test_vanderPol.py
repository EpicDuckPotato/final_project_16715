from pydrake.symbolic import *
import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from verification import *
from vanderPol import *
from controller import *

def main(args=None):
  xgoal = np.array([0, 0])
  ugoal = np.array([0])
  dxg = np.zeros_like(xgoal)

  model = VanderPol()
  n, m = model.get_dim()
  # Infinite-horizon LQR
  A, B = model.lin_dynamics(xgoal, ugoal)
  Q = np.eye(n)
  R = np.eye(m)
  S = solve_continuous_are(A, B, Q, R)
  K = np.linalg.solve(R, B.transpose()@S)
  lqr_policy = LQRPolicy(xgoal, ugoal, S, K)

  rho = find_roa(model, lqr_policy)

if __name__ == '__main__':
  main()