import numpy as np
import matplotlib.pyplot as plt
from pydrake.symbolic import *

def sample_isocontours(f, xvars, num_samples, std=1):
  nx = len(xvars)
  samples = []
  for i in range(num_samples):
    # Search direction
    alpha = np.random.normal(size=(nx,), scale=std)
    beta = np.random.normal(size=(nx,), scale=std)
    t = MakeVectorContinuousVariable(1, 't')[0]
    subs_dict = {xvars[j]: alpha[j]*t + beta[j] for j in range(nx)}
    f_t = Polynomial(f.Substitute(subs_dict))
    monom_to_coeff = f_t.monomial_to_coefficient_map()
    coeffs = []
    for j in range(f_t.TotalDegree() + 1):
      p = Monomial(t, j)
      if p in monom_to_coeff:
        coeffs.append(monom_to_coeff[p].Evaluate())
      else:
        coeffs.append(0)

    roots = np.polynomial.polynomial.polyroots(coeffs)
    for root in roots:
      if np.abs(np.imag(root)) < 1e-5:
        x = alpha*np.real(root) + beta
        if np.linalg.norm(x, ord=np.inf) < 100:
          samples.append(x)
          # print(f.Substitute({xvars[j]: x[j] for j in range(nx)}))

  return samples
