import numpy as np
import matplotlib.pyplot as plt

def plot_ellipse(S, rho, x0, f):
  theta = np.linspace(0, 2*np.pi, 100)
  points = np.sqrt(rho)*np.stack((np.cos(theta), np.sin(theta)), 0)
  L = np.linalg.cholesky(S)
  points = np.linalg.solve(L.transpose(), points)

  for point in points.transpose():
    print(f(point))

  for i in range(2):
    points[i] += x0[i]

  plt.plot(points[0], points[1], linewidth=6)
  plt.show()
