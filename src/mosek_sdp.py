import mosek
from   mosek.fusion import *

def main(args=None):
  with Model("sdo1") as M:

    # Setting up the variables
    X = M.variable("X", Domain.inPSDCone(3))
    x = M.variable("x", Domain.inQCone(3))

    # Setting up constant coefficient matrices
    C  = Matrix.dense ( [[2.,1.,0.],[1.,2.,1.],[0.,1.,2.]] )
    A1 = Matrix.eye(3)
    A2 = Matrix.ones(3,3)

    # Objective
    M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(C, X), x.index(0)))

    # Constraints
    M.constraint("c1", Expr.add(Expr.dot(A1, X), x.index(0)), Domain.equalsTo(1.0))
    M.constraint("c2", Expr.add(Expr.dot(A2, X), Expr.sum(x.slice(1,3))), Domain.equalsTo(0.5))

    M.solve()

    print(X.level())
    print(x.level())

if __name__ == '__main__':
  main()
