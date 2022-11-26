import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt
import sys
from pydrake.symbolic import *
from pydrake.all import (Polynomial, Variable, Evaluate, Substitute,
                         MathematicalProgram, MosekSolver)

sys.path.append('src')
from n_link_cartpole_sympy import *
from controller import *

def main(args=None):
  N = 3 # Number of revolute joints. There is always one prismatic joint, controlling the cart
  link_length = 1
  link_mass = 1

  model = NLinkCartpole(N, link_length, link_mass)
  n, m = model.get_dim()

  xgoal = np.zeros(n)
  Strig, Ktrig, _, _ = model.trig_lqr()

  # Stabilize from some perturbed initial condition
  dt = 0.01
  K = 1000
  thist = dt*np.linspace(0, K)
  xhist = np.zeros((n, K))
  xhist[:, 0] = np.copy(xgoal)
  xhist[0, 0] += 0.5 # Start with perturbed cart position
  xhist[1, 0] += 0.5 # Start with perturbed first joint angle

  for k in range(K-1):
    x_trig = np.zeros(model.nx)
    x_trig[0] = xhist[0, k]
    x_trig[1:1 + N] = np.cos(xhist[1:model.nq_minimal, k])
    x_trig[1 + N:model.nq] = np.sin(xhist[1:model.nq_minimal, k])
    x_trig[model.nq:] = xhist[model.nq_minimal:, k]
    uk = -Ktrig@x_trig
    temp = integrate(xhist[:, k], uk, model.dynamics, dt)
    xhist[:, k+1] = np.copy(temp)
  
  plt.plot(xhist[0,:],xhist[1,:])
  plt.plot(xgoal[0], xgoal[1], color='g', marker="*", markersize=10)
  plt.show()

if __name__ == '__main__':
  main()
