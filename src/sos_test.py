from pydrake.symbolic import *
import numpy as np
from verification import *

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

w = MakeVectorContinuousVariable(2, 'w')
x = w[0]
y = w[1]
f = Polynomial(4*x**4 + 4*x**3*y - 7*x**2*y**2 - 2*x*y**3 + 10*y**4) # This should give True
is_sos = check_sos(f, w)
print(is_sos)
f = Polynomial(x**2 - 1) # This should give False
is_sos = check_sos(f, w)
print(is_sos)
