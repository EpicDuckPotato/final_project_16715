from pydrake.symbolic import *
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian
import numpy as np
import sys
sys.path.append('src')

from verification import *

def verify(gen_V, gen_Vdot, size_w, deg_la, rho, eps):
  w = MakeVectorContinuousVariable(size_w, 'w')
  V = gen_V(w)
  Vdot = gen_Vdot(w)
  return check_sos(-Vdot - eps*Polynomial(w@w), w, [rho - V], [deg_la], [True])

def verify_drake(gen_V, gen_Vdot, size_w, deg_la, rho, eps):
  prog = MathematicalProgram()
  w = prog.NewIndeterminates(size_w, 'w')
  V = gen_V(w).ToExpression()
  Vdot = gen_Vdot(w).ToExpression()
  la = prog.NewSosPolynomial(Variables(w), deg_la)[0].ToExpression()
  prog.AddSosConstraint(-Vdot - la*(rho - V) - eps*w@w)
  result = Solve(prog)
  return result.is_success()

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

def gen_V(w):
  x = w[0]
  y = w[1]
  return Polynomial(x**2 + y**2) 

def gen_Vdot(w):
  x = w[0]
  y = w[1]
  return Polynomial(x**2 + y**2 - 25) # x\dot = Ax with eig(A) = -0.5, -0.5

size_w = 2
deg_la = 2
eps = 0.001
print('Testing first Lyapunov function')
for rho in range(20,30):
  is_sos_drake = verify_drake(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  is_sos = verify(gen_V, gen_Vdot, size_w, deg_la, rho, eps)
  print(f'rho = {rho:.3f} Our solution: {is_sos:1}, Drake solution: {is_sos_drake:1}')

w = MakeVectorContinuousVariable(size_w, 'w')
rho = check_sos_sample(gen_V(w), gen_Vdot(w), w)
print(f'Quotient-Ring SOS: rho = {rho:.4f}')
