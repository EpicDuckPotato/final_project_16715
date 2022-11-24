'''CONTROLLER AND ROA FINDING

'''

import numpy as np
from pydrake.symbolic import *
from verification import *
import matplotlib.pyplot as plt

class ZeroPolicy(object):
  def __init__(self, nu):
    self.nu = nu

  def get_u(self, x, t=0):
    return np.zeros(self.nu)
  
  def get_xg(self, t=0):
    return np.zeros(2)
  
  def get_K(self, t=0):
    return np.zeros(2)
  

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
  print('Finished original ROA search with rho = %f' %(rho))
  return rho


def find_passive_roa(model, MAX_ITER=50):
  deg_Taylor = 3  # order of Taylor expansion of xerrdot
  dxg = np.array([0, 0])  # derivative at x goal
  n, m = model.get_dim()
  # Construct Lyapunov function: V and dV
  xerr = MakeVectorContinuousVariable(n, 'xerr')
  xerrdot = model.sym_dynamics(xerr, ZeroPolicy(m))
  for i in range(n):
    xerrdot[i] = TaylorExpand(xerrdot[i], {var: 0 for var in xerr}, deg_Taylor) 

  V = Polynomial(model.known_V(xerr))
  dV = Polynomial(sum([V.ToExpression().Differentiate(xerr[i])*xerrdot[i] for i in range(n)]))
  # dV = Polynomial(np.dot(model.known_Vgrad(xerr), xerrdot))
  w = MakeVectorContinuousVariable(n, 'w')
  eps = 0.001 # choose epsilon
  # eps = 0 # choose epsilon
  la_degs = [4]  # choose degree of lambdas
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

    # print('ROA search iteration %d, testing rho = %f' %(i, rho))
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
  

def find_passive_roa_sample(model, xlb, xub):
  deg_Taylor = 3  # order of Taylor expansion of xerrdot
  dxg = np.array([0, 0])  # derivative at x goal
  n, m = model.get_dim()
  # Construct Lyapunov function: V and dV
  xerr = MakeVectorContinuousVariable(n, 'xerr')
  xerrdot = model.sym_dynamics(xerr, ZeroPolicy(m)) - dxg
  for i in range(n):
    xerrdot[i] = TaylorExpand(xerrdot[i], {var: 0 for var in xerr}, deg_Taylor) 

  V = Polynomial(model.known_V(xerr))
  # dV = Polynomial(np.dot(model.known_Vgrad(xerr), xerrdot))
  dV = Polynomial(sum([V.ToExpression().Differentiate(xerr[i])*xerrdot[i] for i in range(n)]))

  rho = check_sos_sample(V, dV, xerr, xlb, xub)
  print('Finished quotient-ring ROA search with rho = %f' %(rho))
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
  print('Finished quotient-ring ROA search with rho = %f' %(rho))
  return rho


def find_roa_simulation_2d(model, policy):
  n, m = model.get_dim()
  dt = 0.1
  N = 200
  steps = 200
  xs_eval = np.zeros((n, N))

  if type(model).__name__ == 'Pendulum':
    xs_eval[0,:] = np.linspace(0, 2*np.pi, N)
    xs_eval[1,:] = np.linspace(-2*np.pi, 2*np.pi, N)

  if type(model).__name__ == 'TimeReversedVanderPol':
    xs_eval[0,:] = np.linspace(-2.1, 2.1, N)
    xs_eval[1,:] = np.linspace(-3, 3, N)

  # X,Y = np.meshgrid(xs_eval[0,:], xs_eval[1,:])
  # x_grid = []
  # u = 
  
  # for step in range(steps):
  #   for i in range(N):
  #     for j in range(N):
  #       x = [X[i,j], Y[i,j]]
  #       uk = policy.get_u(x)
  #       temp = integrate(x, uk, model.dynamics, dt)
  #       x = np.copy(temp)
  #       if np.linalg.norm(x-policy.get_xg()) < 1e-3:
  #         x_grid.append([X[i,j], Y[i,j]])

  # plt.plot(x_grid[0], x_grid[1])
  # plt.show()

  x = np.stack(np.meshgrid(xs_eval[0,:], xs_eval[1,:]), 0)
  xd = policy.get_xg()
  xd = np.tile(xd.reshape(n, 1, 1), (1, N, N))
  u = np.zeros((N, N))
  K = policy.get_K()  

  for step in range(steps):
    # if step%N == 0:
      # print('Simulation step %d' %(step))
    errs = x - xd
    for row in range(N):
      for col in range(N):
        u[row, col] = -K@errs[:, row, col]
    x = integrate(x, u, model.dynamics, dt) 

  errs = x - xd
  stable_idx = np.logical_and(np.abs(errs[0]) < 0.01, 
                              np.abs(errs[1]) < 0.01)

  image = np.zeros((N, N, 3))
  image[stable_idx] = 1

  plt.imshow(image, extent=[np.min(xs_eval[0,:]), np.max(xs_eval[0,:]), 
            np.min(xs_eval[1,:]), np.max(xs_eval[1,:])], 
            aspect='auto', origin='lower')
  
  plt.scatter(N, N, color='white', label='Simulation-based ROA Estimate')
  plt.xlim(np.min(xs_eval[0,:]), np.max(xs_eval[0,:]))
  plt.ylim(np.min(xs_eval[1,:]), np.max(xs_eval[1,:]))
  plt.legend() 
  plt.savefig('conservativity.png')
  plt.show()

  # to draw limit cycle, need V=rho or Vdot=0
  return 0


