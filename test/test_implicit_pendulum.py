import numpy as np
from scipy.linalg import solve_continuous_are, expm
import matplotlib.pyplot as plt
import sys

sys.path.append('src')
from implicit_pendulum import Pendulum
from controller import *

def main(args=None):
  model = Pendulum()

  find_lqr_roa_implicit(model)
  # find_lqr_roa_implicit_sample(model)

if __name__ == '__main__':
  main()
