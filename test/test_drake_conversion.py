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

  q_pris = MakeVectorContinuousVariable(1, 'q_pris')[0]
  c_rev = MakeVectorContinuousVariable(N, 'c')
  s_rev = MakeVectorContinuousVariable(N, 's')
  v = MakeVectorContinuousVariable(1 + N, 'v')
  vdot = MakeVectorContinuousVariable(1 + N, 'vdot')
  u = MakeVectorContinuousVariable(N, 'u')

  constraints = model.get_drake_constraints(q_pris, c_rev, s_rev, v, vdot, u)

  # Try out random inputs, make sure that the result of evaluating the sympy dynamics satisfies
  # the drake constraints
  x_vals = np.random.normal(size=(n,))
  u_vals = np.random.normal(size=(m,))
  xdot_vals = model.dynamics(x_vals, u_vals)

  q_vals = x_vals[:n//2]
  v_vals = x_vals[n//2:]
  vdot_vals = xdot_vals[n//2:]
  c_vals = [np.cos(qi) for qi in q_vals[1:]]
  s_vals = [np.sin(qi) for qi in q_vals[1:]]
  for constraint in constraints:
    subs_dict = {q_pris: q_vals[0]}
    subs_dict.update({c: c_val for c, c_val in zip(c_rev, c_vals)})
    subs_dict.update({s: s_val for s, s_val in zip(s_rev, s_vals)})
    subs_dict.update({vi: v_val for vi, v_val in zip(v, v_vals)})
    subs_dict.update({vdoti: vdot_val for vdoti, vdot_val in zip(vdot, vdot_vals)})
    subs_dict.update({ui: u_val for ui, u_val in zip(u, u_vals)})
    print(constraint.ToExpression().Substitute(subs_dict).Evaluate())

if __name__ == '__main__':
  main()
