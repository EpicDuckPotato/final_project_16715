"""Summary
x: sampled state space or indeterminates
Y = [x, V, xxd, trans_psi, T] are all that's necessary for the SDP
T: transformation matrix due to the coordiante ring, serves to reduce the
dimension of the monomial_basis down
P: r.h.s. SOS decomposition Gram matrix
rho: scalar Lyapunov level
"""

import mosek
from mosek.fusion import *
from pydrake.symbolic import *
from pydrake.all import (Polynomial, Variable, Evaluate, Substitute,
                         MathematicalProgram, MosekSolver)
from itertools import combinations_with_replacement
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)


# check_sos: finds a set of Lagrange multiplier polynomials [la_1, la_2, ..., la_n]
# to make the total polynomial L = poly_0 - la_1*poly_1 - ... la_n*poly_n SOS
# ARGUMENTS
# poly0, object of class pydrake.symbolic.Polynomial
# w: a numpy array, where each element of the array is a pydrake.symbolic.Variable.
# constraint_poly: list of constraint polynomials [poly_1, ... poly_n]
# la_degrees: list of degrees for Lagrange multiplier polynomials [deg_la_1, ... deg_la_n]
# la_sos: list indicating whether this multiplier should be SOS. If the constraint
# is an inequality constraint, it should be SOS. Otherwise, it shouldn't be.
# All polynomials should be functions of w
# RETURN: True/False indicating whether we could find a set of
# Lagrange multiplier polynomials to make the total polynomial L SOS
def check_sos(poly0, w, constraint_poly=[], la_degrees=[], la_sos=[]):
	# TODO: this only handles SOS multipliers for now, so we couldn't do equality constraints
	multipliers = [] # List of multiplier polynomials (lambdas)
	multiplier_Qs = [] # Matrix for each lambda
	num_multiplier_basis = [] # Basis size for each lambda
	for i, (deg_la, poly) in enumerate(zip(la_degrees, constraint_poly)):
		# Build basis of each Lagrange multiplier, starting from bias value 1
		basis_la = get_basis(w, deg_la)
		
		num_basis_la = len(basis_la)
		num_multiplier_basis.append(num_basis_la)
		Q_la = MakeMatrixContinuousVariable(num_basis_la, num_basis_la,'Q_la' + str(i))
		multiplier_Qs.append(Q_la)
		multipliers.append(Polynomial(basis_la@Q_la@basis_la, w))

	# Build basis of L, starting from bias value 1
	deg_L = np.max([poly0.TotalDegree()] + [deg_la + poly.TotalDegree() for deg_la, poly in zip(la_degrees, constraint_poly)])
	if deg_L%2 == 1:
		# If the polynomial is odd, it can't be SOS, so we automatically return False
		return False  
	
	basis = get_basis(w, deg_L)
	num_basis = len(basis)
	Q_L = MakeMatrixContinuousVariable(num_basis, num_basis, 'Q_L')
	L = poly0 - sum([la*poly for la, poly in zip(multipliers, constraint_poly)])
	wQw = Polynomial(basis@Q_L@basis, w)

	# Dicts where keys are monomials, values are coefficients
	mon2coef_L_dict = L.monomial_to_coefficient_map()
	mon2coef_wQw_dict = wQw.monomial_to_coefficient_map()   

	with Model("sdo1") as M:
		# PSD matrix variable for LHS
		Q_L_mosek = M.variable(Domain.inPSDCone(num_basis))

		# PSD matrix variables for RHS lambdas
		multiplier_Qs_mosek = []

		for num_basis_la in num_multiplier_basis:
			multiplier_Qs_mosek.append(M.variable(Domain.inPSDCone(num_basis_la)))

		# Now we set up constraints such that the coefficients of w@Q@w equal the coefficients of L
		cidx = 0
		for monom, coef in mon2coef_wQw_dict.items():
			# coeff is what multiplies the monomial in w@Q@w. It's a linear combination of elements of Q, e.g.
			# a00*Q[0, 0] + a01*Q[1, 1] + ...
			# Here, we get the coefficients a00, a01 to set up the LHS of our constraint.
			C_Q_L = np.reshape([coef.Differentiate(q).Evaluate() for q in Q_L.flatten()], Q_L.shape)  

			# Now we get the RHS, i.e. const + a00*Qla[0, 0] + a01*Qla[1, 1] + ... = RHS 
			C_multipliers = []
			const = 0.0
			if monom in mon2coef_L_dict:
				coef_L = mon2coef_L_dict[monom]
				for Q_la in multiplier_Qs:
					C_Q_la = np.reshape([coef_L.Differentiate(q).Evaluate() for q in Q_la.flatten()], Q_la.shape)  
					C_multipliers.append(C_Q_la)

				mon2coef_this_dict = Polynomial(coef_L).monomial_to_coefficient_map()
				if Monomial() in mon2coef_this_dict:
					const = mon2coef_this_dict[Monomial()]
					const = const.Evaluate()

			# Constraints: equalize the coefficients
			LHS = Expr.dot(Matrix.dense(C_Q_L), Q_L_mosek)
			RHS = Expr.constTerm(const)
			for C_Q_la, Q_la_mosek in zip(C_multipliers, multiplier_Qs_mosek):
				RHS = Expr.add(RHS, Expr.dot(Matrix.dense(C_Q_la), Q_la_mosek))
			M.constraint(Expr.sub(LHS, RHS), Domain.equalsTo(0))

			# TODO: optimize code
			# do not have to evaluate full matrix because Qmosek is symmetric

			cidx += 1
		M.setLogHandler(sys.stdout)            # Add logging       
		M.writeTask("original.ptf")                # Save problem in readable format
		M.solve()

		status = M.getPrimalSolutionStatus()

		if status == SolutionStatus.Optimal:
			return True
		else:
			return False


# Generate monomials                
def get_basis(w, deg):
	# w is variables
	# deg is degree of polynomial
	basis = [Expression(1)] 
	for d in range(1, deg//2 + 1):
		combs = combinations_with_replacement([Expression(wi) for wi in w], r=d) # Unordered
		for p in combs:
			basis.append(np.prod(p))
	return basis


# Check SOS with sampling method
def check_sos_sample(sym_V, sym_Vdot, w, xlb=-100, xub=100):
  # Step 1: get samples from Vdot(x) = 0
  samples = []
  num_samples = 5
  # num_samples = 100
  for i in range(num_samples):
    if len(samples) >= num_samples:
      break
    samples.extend(sample_isocontours(sym_Vdot.ToExpression(), w, num_samples, xlb, xub, std=1))
    # samples.extend(sample_vector_isocontours(np.array([sym_Vdot.ToExpression()]), w, num_samples, std=1))

  d = 1
  deg = sym_V.TotalDegree() + 2*d
  
  # Get V(xi) values
  enough_sample = False
  while not enough_sample:
    samples.extend(sample_isocontours(sym_Vdot.ToExpression(), w, 1, xlb, xub, std=1))
    V = np.array([sym_V.Evaluate(dict(zip(w, xi))) for xi in samples])
    samples, V = balancing_V(samples, V)
    xxd, psi = get_sample_features(w, deg, d, samples)
    enough_sample = check_genericity(psi) 

  samples = np.array(samples)
  print(f"Number of samples: {len(samples)}")  
  # plt.scatter(samples[:, 0], samples[:, 1])
  # plt.xlim(-2.1, 2.1)
  # plt.ylim(-3, 3)
  # plt.show()
  # Step 2: solve SDP on samples

  rho = solve_SDP_samples(V, psi, xxd)

  return rho


# Solve SDP with samples
def solve_SDP_samples(V, basis, xxd):
  with Model("sdo2") as M:
    rho = M.variable(Domain.greaterThan(0))
    num_basis = basis.shape[1]
    P = M.variable(Domain.inPSDCone(num_basis))

    for i in range(basis.shape[0]):
      # r = xxd[i] * (V[i] - rho) - basis[i].T@P@basis[i]
      term1 = Expr.mul(xxd[i], Expr.sub(V[i], rho))
      term2 = Expr.mul(Expr.mul(basis[i].reshape(1,num_basis), P), 
      								basis[i].reshape(num_basis,1))
      r = Expr.sub(term1, term2)
      M.constraint(r, Domain.equalsTo(0))

    M.objective(ObjectiveSense.Maximize, rho)
    M.setLogHandler(sys.stdout)            # Add logging
    M.writeTask("sample.ptf")                # Save problem in readable format
    M.solve()

    status = M.getPrimalSolutionStatus()
    if status == SolutionStatus.Unknown:
         print('Mosek cannot solve with these samples. Please run again or increase the number of samples.')
         return 0.0
    
    P_sol = P.level()
    rho_sol = rho.level()[0]

  return rho_sol


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


# Get samples of Vdot(x) = 0 to feed into SDP
def sample_isocontours(f, xvars, num_samples, xlb, xub, std=1):
  np.random.seed(0)
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
        if np.all(x >= xlb) and np.all(x <= xub):
          samples.append(x)
          # print(f.Substitute({xvars[j]: x[j] for j in range(nx)}))

  return samples


# f should now be a vector of Drake expressions. This applies Newton's method
def sample_vector_isocontours(f, xvars, num_samples, std=1):
  nx = len(xvars)
  samples = []
  for i in range(num_samples):
    # Randomly sample an initial point
    x = np.random.normal(size=(nx,), scale=std)
    max_newton_iter = 100
    success = False

    # Run Newton
    for j in range(max_newton_iter):
      subs_dict = {xvars[k]: x[k] for k in range(nx)}
      r = np.array([fk.Substitute(subs_dict).Evaluate() for fk in f])

      # Stopping condition
      if 0.5*r@r < 1e-10:
        success = True
        break

      # Search direction
      J = np.array([[fk.Differentiate(xl).Substitute(subs_dict).Evaluate() for xl in xvars] for fk in f])
      delta_x, _, _, _ = np.linalg.lstsq(J, -r, rcond=-1)
      
      # Line search
      max_line_search = 10
      alpha = 1
      b = 0.1
      for m in range(max_line_search):
        subs_dict = {xvars[k]: x[k] + delta_x[k] for k in range(nx)}
        rcand = np.array([fk.Substitute(subs_dict).Evaluate() for fk in f])
        # Armijo
        if 0.5*rcand@rcand < 0.5*r@r + b*alpha*r.transpose()@J@delta_x:
          x += delta_x
          break

        alpha *= 0.5

    if success:
      samples.append(x)

  return samples

# Taken from https://github.com/shensquared/veril
def balancing_V(x, V, tol=5):
  balanced = np.max(V) / np.min(V) < tol
  while not balanced:
    # print('not balanced')
    idx = [np.argmax(V), np.argmin(V)]
    # test_x = np.vstack([test_x, x[idx]])
    x = (np.delete(x, idx, axis=0)).tolist()
    V = np.delete(V, idx, axis=0)
    balanced = np.max(V) / np.min(V) < tol
  return x, V


# Taken from https://github.com/shensquared/veril
def get_sample_features(w, deg, d, samples):
  basis = get_basis(w, deg)
  num_basis = len(basis)
  psi = np.zeros((num_basis,))
  xxd = [0]

  for xi in samples:
    this_basis = np.array([this.Evaluate(dict(zip(w, xi))) for this in basis])
    this_xxd = (np.array(xi)@np.array(xi))**d
    psi = np.vstack([psi, this_basis])
    xxd.append(this_xxd)
  	
  psi = psi[1:]
  xxd = xxd[1:]    
  return xxd, psi


# Taken from https://github.com/shensquared/veril
def check_genericity(psi):
  enough_samples = True
  m, n = psi.shape
  n2 = n * (n + 1) / 2
  m0 = np.min([m, n2])
  # sub_samples = psi[:m0, :]
  sub_samples = psi

  c = np.power(sub_samples@sub_samples.T, 2)  # c = q'*q
  # print('c shape is %s' % str(c.shape))
  s = np.abs(np.linalg.eig(c)[0])
  tol = np.max(c.shape) * np.spacing(np.max(s)) * 1e3
  sample_rank = sum(s > tol)
  # print('sample rank is %s' % sample_rank)
  if sample_rank == m0 and sample_rank < n2:
    # meaning m<n2 and sample full rank
    # print('Insufficient samples!!')
    enough_samples = False
  return enough_samples
