import numpy as np

# f should be a vector-valued function of multiple variables, returning a 1D numpy array.
# If you want to use a scalar function, just have f return a length-1 numpy array.
# J should be the Jacobian of that function (2D numpy array).
# This just applies Newton's method repeatedly. First, it starts from x0. Then
# it moves to some new starting point along the null-space of J and runs Newton again,
# and so on
def sample_isocontours(f, J_fn, x0, num_samples, max_newton_iter=10, alpha=1):
  samples = []
  x = np.copy(x0)
  I = np.eye(x.shape[0])
  for i in range(num_samples):
    if len(samples) > num_samples:
      break

    for j in range(max_newton_iter):
      r = f(x)
      if np.linalg.norm(r, ord=np.inf) < 1e-5:
        samples.append(np.copy(x))
        break

      J = J_fn(x) 
      x -= np.linalg.lstsq(J, r)[0]

    if len(samples) < 2:
      direction = np.random.normal(size=(x.shape[0]))
    else:
      direction = alpha*(x - np.mean(np.array(samples), 0))

    J = J_fn(x) 
    x += (I - np.linalg.lstsq(J, J)[0])@direction

  return np.array(samples)
