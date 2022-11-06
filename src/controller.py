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
  w = MakeVectorContinuousVariable(n, 'w')
  eps = 0.001 # choose epsilon
  la_degs = [2]  # choose degree of lambdas
  la_SOS = [True] # if lambda is SOS

  rho = 5
  rho_prev = 0.0
  TOL_RHO = 1e-3
  lower = 0.0
  i = 0
  # Find a bracket of rho
  is_sos = check_sos(-dV - eps*Polynomial(w@w), xerr, [rho - V], la_degs, la_SOS)
  while is_sos:
    rho = rho*2
    is_sos = check_sos(-dV - eps*Polynomial(w@w), xerr, [rho - V], la_degs, la_SOS)
  upper = rho
  rho = (lower + upper)/2
  # line search
  while (rho - rho_prev) > TOL_RHO:
      
    if i > MAX_ITER:
      break

    print('ROA search iteration %d, testing rho = %f' %(i, rho))
    is_sos = check_sos(-dV - eps*Polynomial(w@w), xerr, [rho - V], la_degs, la_SOS)

    if is_sos:
      lower = rho
    else:
      upper = rho

    rho_prev = lower
    rho = (lower + upper)/2
    i += 1

  if lower == 0:
    print('No region of attraction')

  rho = lower
  print('Finished ROA search with rho = %f' %(rho))
  return rho
  

def find_roa_sample(model, policy):
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

  rho = check_sos_sample(V, dV, xerr)
  print('Finished ROA search with rho = %f' %(rho))
  return rho