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
  return Polynomial(0.5*x**2 + 0.5*y**2) 

def gen_Vdot(w):
  x = w[0]
  y = w[1]
  return Polynomial(0.5*x**2 + 0.5*y**2 - 1.5) # x\dot = Ax with eig(A) = -0.5, -0.5

size_w = 2
deg_la = 2
eps = 0.001
print('Testing first Lyapunov function')
for rho in np.linspace(0.1, 3, 20):
  is_sos_drake = verify_drake(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  is_sos = verify(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  print(f'rho = {rho:.3f} Our solution: {is_sos:1}, Drake solution: {is_sos_drake:1}')

def gen_V(w):
  return Polynomial(1.6131259297527509*w[1]**2 + 4.8284271247461827*w[0] * w[1] + 4.6955181300451407*w[0]**2)

def gen_Vdot(w):
  return Polynomial(-3.9510190129866094e-16*w[1] + -3.6021752652406729*w[1]**2 + -5.9131178232366702e-16*w[0] + -7.7888609948495748*w[0] * w[1] + -6.8284271247461623*w[0]**2 + 1.9755095064933047e-16*w[0]**2 * w[1] + 2.9565589116183351e-16*w[0]**3 + -0.53770864325091694*w[0]**3 * w[1] + -0.80473785412436372*w[0]**4)

size_w = 2
deg_la = 2
eps = 0.001
print('Testing second Lyapunov function')
for rho in np.linspace(0.05, 0.15, 20):
  is_sos_drake = verify_drake(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  is_sos = verify(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  print(f'rho = {rho:.3f} Our solution: {is_sos:1}, Drake solution: {is_sos_drake:1}')
