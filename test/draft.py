from pydrake.symbolic import *
import numpy as np
x = np.array([[1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [0.5, 0.866],
            [0.5, -0.866]])
n = x.shape[0]
a = np.eye(n)
a = np.vstack([a, a])
1==1