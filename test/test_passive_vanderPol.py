from pydrake.symbolic import *
import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt
import sys
sys.path.append('src')

from verification import *
from time_reversed_vanderPol import *
from controller import *

def main(args=None):
  model = TimeReversedVanderPol()
  n, m = model.get_dim()

  rho = find_passive_roa(model)
  rho = find_passive_roa_sample(model, -3, 3)
  # rho2 = find_roa_simulation_2d(model, ZeroPolicy(m))  

if __name__ == '__main__':
  main()
