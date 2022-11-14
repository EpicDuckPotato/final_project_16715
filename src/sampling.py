import numpy as np
import matplotlib.pyplot as plt

# f should be a scalar-valued function, and grad_fn should give its gradient wrt x.
# This just applies Newton's method repeatedly in random directions, i.e.
# we pick a direction x = alpha*t + beta, then find roots along this line
def sample_isocontours(f, grad_fn, nx, num_samples, x0, std=1, max_newton_iter=10):
  samples = []
  for i in range(num_samples):
    alpha = np.random.normal(size=(nx,))
    beta = np.random.normal(size=(nx,))
    t = std*np.random.normal()
    for j in range(max_newton_iter):
      x = alpha*t + beta + x0
      r = f(x)
      if abs(r) < 1e-5:
        samples.append(np.copy(x))
        break

      dr_dt = grad_fn(x)@alpha
      t -= r/dr_dt

  return samples
