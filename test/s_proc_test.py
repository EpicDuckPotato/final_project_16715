from pydrake.symbolic import *
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian
import numpy as np
import sys
sys.path.append('src')

from verification import *

def verify_drake(gen_V, gen_Vdot, size_w, deg_la, rho, eps):
  prog = MathematicalProgram()
  w = prog.NewIndeterminates(size_w, 'w')
  V = gen_V(w).ToExpression()
  Vdot = gen_Vdot(w).ToExpression()
  la = prog.NewSosPolynomial(Variables(w), deg_la)[0].ToExpression()
  prog.AddSosConstraint(-Vdot - la*(rho - V) - eps*w@w)
  result = Solve(prog)
  return result.is_success()

def verify(gen_V, gen_Vdot, size_w, deg_la, rho, eps):
  w = MakeVectorContinuousVariable(size_w, 'w')
  V = gen_V(w)
  Vdot = gen_Vdot(w)
  return check_sos(-Vdot - eps*Polynomial(w@w), w, [rho - V], [deg_la], [True])

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

def gen_V(w):
  x = w[0]
  y = w[1]
  return Polynomial(x**2 + y**2) 

def gen_Vdot(w):
  x = w[0]
  y = w[1]
  return Polynomial(x**2 + y**2 - 1) # x\dot = Ax with eig(A) = -0.5, -0.5

size_w = 2
deg_la = 2
eps = 0.001
print('Testing first Lyapunov function')
for rho in np.linspace(0.1, 3, 20):
  is_sos_drake = verify_drake(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  is_sos = verify(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  print(f'rho = {rho:.3f} Our solution: {is_sos:1}, Drake solution: {is_sos_drake:1}')

def gen_V(w):
  x0 = w[0]
  x1 = w[1]
  V = (1.8027e-06) + (0.28557) * x1**2 + (0.0085754) * x1**4 + \
      (0.18442) * x0**2 + (0.016538) * x0**4 + \
      (-0.34562) * x0 * x1 + (0.064721) * x0 * x1**3 + \
      (0.10556) * x0**2 * x1**2 + (-0.060367) * x0**3 * x1
  return V


def gen_Vdot(w):
  x0 = w[0]
  x1 = w[1]
  Vgrad = np.array([2*(0.18442) * x0 + 4*(0.016538) * x0**3 + (-0.34562) * x1 + (0.064721) * x1**3 + 2*(0.10556) * x0 * x1**2 + 3*(-0.060367) * x0**2 * x1, \
                    2*(0.28557) * x1 + 4*(0.0085754) * x1**3 + (-0.34562) * x0 + 3*(0.064721) * x0 * x1**2 + 2*(0.10556) * x0**2 * x1 + (-0.060367) * x0**3])
  return Polynomial(np.dot(model.Vgrad(xerr), xerrdot))

size_w = 2
deg_la = 2
eps = 0.001
print('Testing second Lyapunov function')
for rho in np.linspace(0.05, 0.15, 20):
  is_sos_drake = verify_drake(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  is_sos = verify(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  print(f'rho = {rho:.3f} Our solution: {is_sos:1}, Drake solution: {is_sos_drake:1}')
