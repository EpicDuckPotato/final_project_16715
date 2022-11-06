import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt
import sys

sys.path.append('src')
from n_link_cartpole import *
from controller import *

def main(args=None):
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
  lqr_policy = LQRPolicy(xgoal, ugoal, S, K)

  # Stabilize from some perturbed initial condition
  dt = 0.01
  K = 1000
  thist = dt*np.linspace(0, K)
  xhist = np.zeros((nx, K))
  xhist[:, 0] = np.copy(xgoal)
  xhist[0, 0] += 0.5 # Start with perturbed cart position
  xhist[1, 0] += 0.5 # Start with perturbed first joint angle

  for k in range(K-1):
    uk = lqr_policy.get_u(xhist[:, k])
    temp = integrate(xhist[:, k], uk, model.dynamics, dt)
    xhist[:, k+1] = np.copy(temp)
  
  plt.plot(xhist[0,:],xhist[1,:])
  plt.plot(xgoal[0], xgoal[1], color='g', marker="*", markersize=10)
  plt.show()

if __name__ == '__main__':
  main()
