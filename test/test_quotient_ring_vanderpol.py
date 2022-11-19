import sys
sys.path.append('src')
import numpy as np
import matplotlib.pyplot as plt
from controller import *
from verification import *
from scipy.linalg import solve_continuous_are, expm
from vanderPol import *

def main(args=None):
  # Test on vanderpol
  xgoal = np.array([0, 0])
  ugoal = np.array([0])

  model = VanderPol()
  nx, nu = model.get_dim()
  # Infinite-horizon LQR
  A, B = model.lin_dynamics(xgoal, ugoal)
  Q = np.eye(nx)
  R = np.eye(nu)
  S = solve_continuous_are(A, B, Q, R)
  K = np.linalg.solve(R, B.transpose()@S)
  policy = LQRPolicy(xgoal, ugoal, S, K)

  xerr = MakeVectorContinuousVariable(policy.get_xg().shape[0], 'xerr')
  xerrdot = model.sym_dynamics(xerr + policy.get_xg(), policy)
  for i in range(nx):
    xerrdot[i] = TaylorExpand(xerrdot[i], {var: 0 for var in xerr}, 3) 
  Vdot = Polynomial(2*np.dot(xerr, S@xerrdot))

  samples = []
  num_samples = 100
  for i in range(100):
    if len(samples) >= num_samples:
      break
    samples.extend(sample_isocontours(Vdot.ToExpression(), xerr, num_samples, std=1))
  samples = np.array(samples)
  plt.scatter(samples[:, 0], samples[:, 1])
  plt.show()

  w = MakeVectorContinuousVariable(nx, 'w')
  V = np.array([(x - xgoal)@S@(x - xgoal) for x in samples])
  degV = 2
  rho = check_sos_sample_no_sym(V, w, samples, degV)
  print(rho)

if __name__ == '__main__':
  main()
