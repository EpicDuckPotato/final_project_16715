'''SOS VERIFICATION
'''

import mosek
from mosek.fusion import *
from pydrake.symbolic import *
from itertools import combinations_with_replacement
import numpy as np

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

# check_sos: checks if a polynomial is SOS
# ARGUMENTS
# V, dV: Lyapunov and its derivative, object of class pydrake.symbolic.Polynomial
# rho: ROA to check
# w: a numpy array, where each element of the array is a pydrake.symbolic.Variable.
# deg_lam: degree of lamda(w)
# eps: epsilon
# RETURN: True/False indicating whether the polynomial is SOS
def check_sos(V, dV, rho, w, deg_lam=2, eps=0.001, verbose=False):
  # Build basis of lambda(x), start from bias value 1
  basis_la = [Expression(1)] 
  for d in range(1, deg_lam//2+1):
    combs = combinations_with_replacement(w, r=d) # Unordered
    for p in combs:
      basis_la.append(np.prod(p))
  
  num_basis_la = len(basis_la)
  Q_la = MakeMatrixContinuousVariable(num_basis_la, num_basis_la,'Q_la')
  la = Polynomial(basis_la@Q_la@basis_la, w)  # without w, all variables in m are considered as indeterminates

  # Build basis of Lagrangian-ish, start from bias value 1
  deg_L = np.max([dV.TotalDegree(), deg_lam+V.TotalDegree()])
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
  w2 = Polynomial(w@w)
  L = -dV - eps*w2 + la*(rho - V)
  wQw = Polynomial(basis@Q_L@basis, w)

  # Dicts where keys are monomials, values are coefficients
  mon2coef_L_dict = L.monomial_to_coefficient_map()
  mon2coef_wQw_dict = wQw.monomial_to_coefficient_map()   

  with Model("sdo1") as M:
    Q_L1= M.variable(Domain.inPSDCone(num_basis))
    Q_la1= M.variable(Domain.inPSDCone(num_basis_la))

    # Now we set up constraints such that the coefficients of basis_times_Q equal the coefficients of input_poly
    cidx = 0
    for monom, coef in mon2coef_wQw_dict.items():
      # coeff is what multiplies the monomial. It's a linear combination of elements of Q, e.g.
      # a00*Q[0, 0] + a01*Q[1, 1] + ...
      # Here, we get the coefficients a00, a01 to set up the LHS of our constraint.
      L_QL = np.reshape([coef.Differentiate(q).Evaluate() for q in Q_L.flatten()], Q_L.shape)  

      # Now we get the RHS, i.e. c + a00*Qla[0, 0] + a01*Qla[1, 1] + ... = RHS 
      const = 0.0
      if monom in mon2coef_L_dict:
        # check coef and coef_L, that's what we need to equalize
        coef_L = mon2coef_L_dict[monom]
        mon2coef_this_dict = Polynomial(coef_L).monomial_to_coefficient_map()
        # get the constant before div
        if Monomial() in mon2coef_this_dict:
          const = mon2coef_this_dict[Monomial()]
          const = const.Evaluate()

        L_Qla= np.reshape([coef_L.Differentiate(q).Evaluate() for q in Q_la.flatten()], Q_la.shape)  
      else:
        # If the monomial doesn't have any coefficients in input_poly, that implies the coefficient is zero
        L_Qla = 0.0
      

      # Constraints: equalize the coefficients
      LHS = Expr.dot(Matrix.dense(L_QL), Q_L1)
      RHS = Expr.add(Expr.dot(Matrix.dense(L_Qla), Q_la1), const)
      M.constraint(Expr.sub(LHS, RHS), Domain.equalsTo(0)) # dot returns scalar
      # TODO: optimize code
      # do not have to evaluate full matrix because Qmosek is symmetric
      cidx += 1

    M.solve()

    status = M.getPrimalSolutionStatus()

    if status == SolutionStatus.Optimal:
      # if verbose:
        # print('The input polynomial was the following:')
        # print(input_poly)
        # Qsoln = Qmosek.level().reshape(Q.shape)
        # print('The polynomial is SOS, with the following PSD matrix:')
        # print(Qsoln)
        # print('This corresponds to the following SOS decomposition')
        # L = np.linalg.cholesky(Qsoln)
        # polys = L.transpose()@basis
        # for p, poly in enumerate(polys):
        #   print_str = '(' + str(poly) + ')^2'
        #   if p != len(polys) - 1:
        #     print_str += ' + '
        #   print(print_str)
        # print('The resulting polynomial is the following:')
        # output_poly = Polynomial(basis@Qsoln@basis).RemoveTermsWithSmallCoefficients(1e-7)
        # print(output_poly)
        # print()

      return True
    else:
      return False