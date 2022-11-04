from pydrake.symbolic import *
import numpy as np
x = MakeVectorContinuousVariable(2, 'x')
f = Polynomial(1 + x[0]**2)
dic = f.monomial_to_coefficient_map() 
one = Expression(1)
a = Monomial()
const = dic[a]
a=[1,2]+[3,4]
print(a)