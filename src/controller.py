'''CONTROLLER AND ROA FINDING

'''

import numpy as np
from pydrake.symbolic import *
from verification import *

class LQRPolicy(object):
  def __init__(self, xg, ug, S, K):
    self.xg = np.copy(xg)
    self.ug = np.copy(ug)
    self.S = np.copy(S)
    self.K = np.copy(K)
    self.rho = None

  def inf_horizon(self):
    return True

  def get_K(self, t=0):
    return np.copy(self.K)

  def get_xg(self, t=0):
    return np.copy(self.xg)

  def get_ug(self, t=0):
    return np.copy(self.ug)

  def get_S(self, t=0):
    return np.copy(self.S)

  def set_rho(self, rho):
    self.rho = rho

  def get_rho(self, t=0):
    return self.rho

  def in_roa(self, x):
    if self.rho is None:
      print('Querying region of attraction without setting it first')
      return False

    diff = x - self.xg
    return np.dot(diff, self.S@diff) <= self.rho
    
  def get_u(self, x, t=0):
    return self.ug - self.K@(x - self.xg)

def integrate(x, u, dynamics, dt):  # RK4
  x1 = x
  k1 = dynamics(x1, u)
  x2 = x + dt*k1/2
  k2 = dynamics(x2, u)
  x3 = x + dt*k2/2
  k3 = dynamics(x3, u)
  x4 = x + dt*k3
  k4 = dynamics(x4, u)
  return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)    


def find_roa(model, policy, MAX_ITER=50):
  deg_Taylor = 3  # order of Taylor expansion of xerrdot
  dxg = np.array([0, 0])  # derivative at x goal
  n, m = model.get_dim()
 # Construct Lyapunov function: V and dV
  xerr = MakeVectorContinuousVariable(policy.get_xg().shape[0], 'xerr')
  xerrdot = model.sym_dynamics(xerr + policy.get_xg(), policy) - dxg
  for i in range(n):
    xerrdot[i] = TaylorExpand(xerrdot[i], {var: 0 for var in xerr}, deg_Taylor) 

  S = policy.get_S()
  V = Polynomial(np.dot(xerr, S@xerr))
  dV = Polynomial(2*np.dot(xerr, S@xerrdot))

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
    is_sos = check_sos(V, dV, rho, xerr)

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
  return rho
  