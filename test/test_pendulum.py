from pydrake.symbolic import *
import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt

from verification import *
from pendulum import *
from controller import *

def main(args=None):
  xgoal = np.array([np.pi, 0])
  ugoal = np.array([0])
  dxg = np.zeros_like(xgoal)

  model = Pendulum()
  n, m = model.get_dim()
  # Infinite-horizon LQR
  A, B = model.lin_dynamics(xgoal, ugoal)
  Q = np.eye(n)
  R = np.eye(m)
  S = solve_continuous_are(A, B, Q, R)
  K = np.linalg.solve(R, B.transpose()@S)
  lqr_policy = LQRPolicy(xgoal, ugoal, S, K)

  # Find roa (to make a function)

  xg = xgoal
  policy = lqr_policy
  MAX_ITER = 50

  # Construct Lyapunov function: V and dV
  xerr = MakeVectorContinuousVariable(policy.get_xg().shape[0], 'xerr')
  xerrdot = model.sym_dynamics(xerr + xg, policy) - dxg
  for i in range(n):
    xerrdot[i] = TaylorExpand(xerrdot[i], {var: 0 for var in xerr}, 3) 

  S = policy.get_S()
  V = np.dot(xerr, S@xerr)
  dV = Polynomial(2*np.dot(xerr, S@xerrdot))
  la = MakeMatrixContinuousVariable(num_basis, num_basis, 'Q')

  # Line search
  lower = 0
  upper = 20
  rho = 10
  rho_min = 1e-3
  
  i = 0
  while rho > rho_min:
      
    if i > MAX_ITER and lower != 0:
      break

    print('ROA search iteration %d, testing rho = %f' %(i, rho))
    f = Polynomial(dV - la*(rho - V) - 0.001*xerr@xerr)
    is_sos = check_sos(f, xerr)

    if is_sos:
      lower = rho
      rho = (rho + upper)/2
    else:
      upper = rho
      rho = (rho + lower)/2

    i += 1

  if lower == 0:
    print('No region of attraction')

  rho = lower
  print('Finished ROA search with rho = %f' %(rho))



if __name__ == '__main__':
  main()