'''THIS MODULE ONLY TESTS THE DYNAMICS MODEL AND LQR CONTROLLER
Plot the stabilized trajectory at the end
'''

from pydrake.symbolic import *
import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from verification import *
from pendulum import *
from vanderPol import *
from controller import *
from time_reversed_vanderPol import *

def test_pendulum(args=None):
  xgoal = np.array([np.pi, 0])
  ugoal = np.array([0])

  model = Pendulum()
  n, m = model.get_dim()
  # Infinite-horizon LQR
  A, B = model.lin_dynamics(xgoal, ugoal)
  Q = np.eye(n)
  R = np.eye(m)
  S = solve_continuous_are(A, B, Q, R)
  K = np.linalg.solve(R, B.transpose()@S)
  lqr_policy = LQRPolicy(xgoal, ugoal, S, K)

  dt = 0.1
  N = 100
  thist = dt*np.linspace(0, N)
  xhist = np.zeros((n, N))
  xhist[:, 0] = np.array([0, 0])

  for k in range(N-1):
    uk = lqr_policy.get_u(xhist[:, k])
    temp = integrate(xhist[:, k], uk, model.dynamics, dt)
    xhist[:, k+1] = np.copy(temp)
  
  plt.plot(xhist[0,:],xhist[1,:])
  plt.plot(xgoal[0], xgoal[1], color='g', marker="*", markersize=10)
  plt.show()

def test_vanderPol(args=None):
  xgoal = np.array([0, 0])
  ugoal = np.array([0])

  model = TimeReversedVanderPol()
  n, m = model.get_dim()
  # Infinite-horizon LQR
  A, B = model.lin_dynamics(xgoal, ugoal)
  Q = np.eye(n)
  R = np.eye(m)
  S = solve_continuous_are(A, B, Q, R)
  K = np.linalg.solve(R, B.transpose()@S)
  lqr_policy = LQRPolicy(xgoal, ugoal, S, K)

  dt = 0.1
  N = 200
  thist = dt*np.linspace(0, N)
  xhist = np.zeros((n, N))
  xhist[:, 0] = np.array([1, 2.6])
  print(model.known_V(xhist[:, 0]))
  for k in range(N-1):
    uk = lqr_policy.get_u(xhist[:, k])
    temp = integrate(xhist[:, k], uk, model.dynamics, dt)
    xhist[:, k+1] = np.copy(temp)
  print(xhist[:,N-1])
  plt.plot(xhist[0,:],xhist[1,:])
  plt.plot(xgoal[0], xgoal[1], color='g', marker="*", markersize=10)
  # plt.show()

if __name__ == '__main__':
  test_pendulum()
  # test_vanderPol()