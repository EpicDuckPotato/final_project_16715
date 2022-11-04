from pydrake.symbolic import *
import numpy as np
import sys
sys.path.append('src')

from verification import *

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

w = MakeVectorContinuousVariable(2, 'w')
x = w[0]
y = w[1]
V = Polynomial(0.5*x**2 + 0.5*y**2) 
dV = Polynomial(-0.5*x**2 - 0.5*y**2 + 1.9) # x\dot = Ax with eig(A) = -0.5, -0.5
is_sos = check_sos(V, dV, 2, w, deg_lam=2, eps=0.001, verbose=False)
print(is_sos)

V = Polynomial(0.5*x**2 + 0.5*y**2) 
dV = Polynomial(-0.5*x**2 - 0.5*y**2 + 1.9) # x\dot = Ax with eig(A) = -0.5, -0.5
is_sos = check_sos(V, dV, 0.1, w, deg_lam=2, eps=0.001, verbose=False)
print(is_sos)