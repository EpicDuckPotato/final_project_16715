from pydrake.symbolic import *
import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from verification import *
from vanderPol import *
from controller import *

def main(args=None):
  model = VanderPol(-1)
  n, m = model.get_dim()

  rho = find_passive_roa(model)
  print(rho)

if __name__ == '__main__':
  main()
