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

V = Polynomial(1.6131259297527509*w[1]**2 + 4.8284271247461827*w[0] * w[1] + 4.6955181300451407*w[0]**2)

dV = Polynomial(-3.9510190129866094e-16*w[1] + -3.6021752652406729*w[1]**2 + -5.9131178232366702e-16*w[0] + -7.7888609948495748*w[0] * w[1] + -6.8284271247461623*w[0]**2 + 1.9755095064933047e-16*w[0]**2 * w[1] + 2.9565589116183351e-16*w[0]**3 + -0.53770864325091694*w[0]**3 * w[1] + -0.80473785412436372*w[0]**4)

is_sos = check_sos(V, dV, 0.1, w, deg_lam=2, eps=0.001, verbose=False)
print(is_sos)