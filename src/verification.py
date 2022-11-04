'''SOS VERIFICATION
'''

import mosek
from mosek.fusion import *
from pydrake.symbolic import *
from itertools import combinations_with_replacement
import numpy as np

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
    basis_la = [Expression(1)] 
    for d in range(1, deg_la//2+1):
      combs = combinations_with_replacement(w, r=d) # Unordered
      for p in combs:
        basis_la.append(np.prod(p))
    
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

  basis = [Expression(1)] 
  for d in range(1, deg_L//2 + 1):
    combs = combinations_with_replacement(w, r=d) # Unordered
    for p in combs:
      basis.append(np.prod(p))

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

    M.solve()

    status = M.getPrimalSolutionStatus()

    if status == SolutionStatus.Optimal:
      return True
    else:
      return False
