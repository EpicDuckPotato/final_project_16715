import mosek
from mosek.fusion import *
from pydrake.symbolic import *
from itertools import combinations_with_replacement
import numpy as np

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

# check_sos: checks if a polynomial is SOS
# ARGUMENTS
# input_poly: an object of class pydrake.symbolic.Polynomial, which can be created by passing a polynomial pydrake Expression to the Polynomial constructor
# v: a numpy array, where each element of the array is a pydrake.symbolic.Variable.
# This can be created by pydrake.symbolic.MakeVectorContinuousVariable
# RETURN: True/False indicating whether the polynomial is SOS
def check_sos(input_poly, w, verbose=False):
  if input_poly.TotalDegree()%2 == 1:
    # If the polynomial is odd, it can't be SOS, so we automatically return False
    return False

  # The max degree in our basis is half the degree of the input polynomial
  degree = input_poly.TotalDegree()//2 

  basis = [Expression(1)]
  for d in range(1, degree + 1):
    combs = combinations_with_replacement(w, r=d)
    for p in combs:
      basis.append(np.prod(p))

  num_basis = len(basis)

  Q = MakeMatrixContinuousVariable(num_basis, num_basis, 'Q')
  Qflat = Q.flatten()
  basis_times_Q = Polynomial(basis@Q@basis, w)

  with Model("sdo1") as M:
    Qmosek = M.variable("Q", Domain.inPSDCone(num_basis))

    # Now we set up constraints such that the coefficients of basis_times_Q equal the coefficients of input_poly

    # Dicts where keys are monomials, values are coefficients
    basis_times_Q_coeff_dict = basis_times_Q.monomial_to_coefficient_map()
    input_poly_coeff_dict = input_poly.monomial_to_coefficient_map()
    cidx = 0
    for monom, coeff in basis_times_Q_coeff_dict.items():
      # coeff is what multiplies the monomial. It's a linear combination of elements of Q, e.g.
      # a00*Q[0, 0] + a01*Q[1, 1] + ...
      # Here, we get the coefficients a00, a01 to set up the LHS of our constraint.
      LHS_coeffs = np.reshape([coeff.Differentiate(q).Evaluate() for q in Qflat], Q.shape)

      # Now we get the RHS, i.e. a00*Q[0, 0] + a01*Q[1, 1] + ... = RHS
      if monom in input_poly_coeff_dict:
        RHS = input_poly_coeff_dict[monom].Evaluate()
      else:
        # If the monomial doesn't have any coefficients in input_poly, that implies the coefficient is zero
        RHS = 0.0

      # Constraints
      M.constraint('c' + str(cidx), Expr.dot(Matrix.dense(LHS_coeffs), Qmosek), Domain.equalsTo(RHS))
      cidx += 1

    M.solve()

    status = M.getPrimalSolutionStatus()

    if status == SolutionStatus.Optimal:
      if verbose:
        print('The input polynomial was the following:')
        print(input_poly)
        Qsoln = Qmosek.level().reshape(Q.shape)
        print('The polynomial is SOS, with the following PSD matrix:')
        print(Qsoln)
        print('This corresponds to the following SOS decomposition')
        L = np.linalg.cholesky(Qsoln)
        polys = L.transpose()@basis
        for p, poly in enumerate(polys):
          print_str = '(' + str(poly) + ')^2'
          if p != len(polys) - 1:
            print_str += ' + '
          print(print_str)
        print('The resulting polynomial is the following:')
        output_poly = Polynomial(basis@Qsoln@basis).RemoveTermsWithSmallCoefficients(1e-7)
        print(output_poly)
        print()

      return True
    else:
      return False
