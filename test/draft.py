from pydrake.symbolic import *
import numpy as np
import matplotlib.pyplot as plt
from time_reversed_vanderPol import *
from controller import *

N = 200
model = TimeReversedVanderPol()
n, m = model.get_dim()

# Construct Lyapunov function: V and dV
deg_Taylor = 3  # order of Taylor expansion of xerrdot
xerr = MakeVectorContinuousVariable(n, 'xerr')
xerrdot = model.sym_dynamics(xerr, ZeroPolicy(m))
for i in range(n):
    xerrdot[i] = TaylorExpand(xerrdot[i], 
                            {var: 0 for var in xerr}, deg_Taylor) 

V = Polynomial(model.known_V(xerr))
dV = Polynomial(sum([V.ToExpression().Differentiate(xerr[i])*xerrdot[i] for i in range(n)]))

xs_eval = np.zeros((n, N))
xs_eval[0,:] = np.linspace(-2.1, 2.1, N) 
xs_eval[1,:] = np.linspace(-3, 3, N) 

x1, x2 = np.meshgrid(xs_eval[0,:], xs_eval[1,:])

V = (1.8027e-06) + (0.28557) * x1**2 + (0.0085754) * x1**4 + \
        (0.18442) * x2**2 + (0.016538) * x2**4 + \
        (-0.34562) * x2 * x1 + (0.064721) * x2 * x1**3 + \
        (0.10556) * x2**2 * x1**2 + (-0.060367) * x2**3 * x1

levels = np.array([1, 1.25, 2]) 
fig, ax = plt.subplots()
CS = ax.contour(x1, x2, V, levels)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Contour of known V')
plt.show()