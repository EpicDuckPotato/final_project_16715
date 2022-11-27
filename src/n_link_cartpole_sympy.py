import sympy.physics.mechanics as me
import sympy as sm
import numpy as np
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian, Expression, Monomial
from pydrake.symbolic import *
from itertools import combinations_with_replacement
from scipy.linalg import solve_continuous_are, expm

class NLinkCartpole(object):
  def __init__(self, N, link_length, link_mass):
    self.N = N
    self.nq = 1 + 2*self.N
    self.nv = 1 + self.N
    self.nq_minimal = self.nv
    self.nx = self.nq + self.nv
    self.nx_minimal = self.nq_minimal + self.nv
    self.nu = N

    g = 9.81
    t = sm.symbols('t')

    self.q = me.dynamicsymbols('q:{}'.format(self.nq_minimal)) # Generalized coordinates
    self.qdot = me.dynamicsymbols('qdot:{}'.format(self.nv)) # Generalized velocities
    self.u = me.dynamicsymbols('u:{}'.format(self.nu)) # Input forces (none for the last joint)

    I = me.ReferenceFrame('I') # Inertial reference frame
    O = me.Point('O') # Origin point
    O.set_vel(I, 0)

    P0 = me.Point('P0') # Cart position
    P0.set_pos(O, self.q[0] * I.x) # Set the position of P0
    P0.set_vel(I, self.qdot[0] * I.x) # Set the velocity of P0
    Pa0 = me.Particle('Pa0', P0, link_mass) # Define a particle at P0

    frames = [I] # List to hold the n + 1 frames
    points = [P0] # List to hold the n + 1 points
    particles = [Pa0] # List to hold the n + 1 particles
    forces = [(P0, self.u[0] * I.x - link_mass*g*I.z)] # List to hold the applied forces, including the input forces, u
    kindiffs = [self.q[0].diff(t) - self.qdot[0]] # List to hold kinematic ODE's

    for i in range(N):
      Bi = I.orientnew('B' + str(i), 'Axis', [self.q[i + 1], -I.y]) # Create a new frame
      Bi.set_ang_vel(I, -self.qdot[i + 1] * I.y) # Set angular velocity
      frames.append(Bi) # Add it to the frames list

      Pi = points[-1].locatenew('P' + str(i + 1), link_length * Bi.z) # Create a new point at the end of the link
      Pi.v2pt_theory(points[-1], I, Bi) # Set the velocity (1st argument is the point corresponding to the previous joint, second is the frame in which the velocity is defined, third is the frame in which both points are fixed
      points.append(Pi) # Add it to the points list

      Pai = me.Particle('Pa' + str(i + 1), Pi, link_mass) # Create a new particle at the end of the link
      particles.append(Pai) # Add it to the particles list

      forces.append((Pi, -link_mass * g * I.z)) # Set the force applied at the point
      kindiffs.append(self.q[i + 1].diff(t) - self.qdot[i + 1]) # Define the kinematic ODE:  dq_i / dt - qdot_i = 0
    for i in range(N-1):
      forces.append((frames[i+1], -self.u[i+1]*frames[i].y)) # Tuple indicates that we're applying a torque of u[i + 1]*frames[i].y on frames[i + 1]

    self.kane = me.KanesMethod(I, q_ind=self.q, u_ind=self.qdot, kd_eqs=kindiffs) # Initialize the object
    fr, frstar = self.kane.kanes_equations(particles, forces) # Generate EoM's fr + frstar = 0
    dynamics_vars = self.q + self.qdot + self.u
    self.mass_matrix_fn = sm.lambdify(dynamics_vars, self.kane.mass_matrix)
    self.forcing_fn = sm.lambdify(dynamics_vars, self.kane.forcing)

  def get_dim(self):
    return self.nx_minimal, self.nu

  # u actuates the cart, and all revolute joints except the last (top) one
  def dynamics(self, x, u):
    xdot = np.zeros(self.nx_minimal)
    xdot[:self.nv] = np.copy(x[self.nv:])
    xu = np.concatenate((x, u))
    xdot[self.nv:] = np.linalg.solve(self.mass_matrix_fn(*xu), self.forcing_fn(*xu)).flatten()
    return xdot

  def lin_dynamics(self, x, u):
    equilibrium_dict = dict(zip(self.q + self.qdot, x))
    F_A, F_B, r = self.kane.linearize(new_method=True, op_point=equilibrium_dict, A_and_B=True)
    A = sm.matrix2numpy(F_A, dtype=float)
    B = sm.matrix2numpy(F_B, dtype=float)
    return A, B

  def generate_drake_variables(self):
    q_pris = MakeVectorContinuousVariable(1, 'q_pris')
    c_rev = MakeVectorContinuousVariable(self.N, 'c_rev')
    s_rev = MakeVectorContinuousVariable(self.N, 's_rev')
    v = MakeVectorContinuousVariable(1 + self.N, 'v')
    vdot = MakeVectorContinuousVariable(1 + self.N, 'vdot')
    u = MakeVectorContinuousVariable(self.N, 'u')
    return np.concatenate((q_pris, c_rev, s_rev)), v, vdot, u

  # transform from [qdot; vdot] = T*[v; vdot], where q = [q0, c1, s1, c2, s2, ...]
  def get_T(self, q):
    T = np.zeros((self.nq + self.nv, self.nx_minimal), dtype=Expression)
    T[0, 0] = 1
    for i in range(self.N):
      T[1 + i, 1 + i] = -q[1 + self.N + i] # Deriv of cos = -sin
      T[1 + self.N + i, 1 + i] = q[1 + i] # Deriv of sin = cos
    T[-self.nv:, -self.nv:] = np.eye(self.nv, dtype=Expression)
    return T

  # Converts from sympy to Drake
  def get_drake_constraints(self, q, v, vdot, u):
    # Create list of sympy variables
    cos_vars = [sm.Symbol('c' + str(i + 1)) for i in range(self.N)]
    sin_vars = [sm.Symbol('s' + str(i + 1)) for i in range(self.N)]
    config_vars = [self.q[0]] + cos_vars + sin_vars
    v_vars = self.qdot
    vdot_vars = [sm.Symbol('vdot' + str(i)) for i in range(self.nv)]
    sm_vars = config_vars + v_vars + vdot_vars + self.u

    # Create corresponding list of drake variables
    drake_vars = np.concatenate((q, v, vdot, u))
    var_idx = np.arange(len(drake_vars))

    # We'll be substituting cosines and sines with auxiliary configuration variables
    cos_subs = [(sm.cos(self.q[i + 1]), 'c' + str(i + 1)) for i in range(self.N)]
    sin_subs = [(sm.sin(self.q[i + 1]), 's' + str(i + 1)) for i in range(self.N)]
    trig_subs = cos_subs + sin_subs

    sm_constraints = [-self.kane.forcing[i].subs(trig_subs) for i in range(self.nv)]

    # Get sympy constraints and max degree of any polynomial expression in the dynamics
    max_deg = 0
    for i in range(self.nv):
      for j in range(self.nv):
        sm_constraints[i] += self.kane.mass_matrix[i, j].subs(trig_subs)*vdot_vars[j]

      poly = sm.Poly(sm_constraints[i], sm_vars)
      if poly.total_degree() > max_deg:
        max_deg = poly.total_degree()

    # Create bases for sympy and drake
    drake_basis = []
    sm_basis = []
    v_start = self.nq
    vdot_start = v_start + self.nv
    for d in range(max_deg + 1):
      combs = combinations_with_replacement([idx for idx in var_idx], r=d) # Unordered
      for p in combs:
        powers = [0 for idx in var_idx] 
        valid_monom = True
        for idx in p:
          powers[idx] += 1
          if sum(powers[vdot_start:]) > 1 or (np.any(powers[vdot_start:]) and np.any(powers[v_start:vdot_start])):
            # We know that there are no products of vdots, products of vdots and us,
            # products of vdots and vs, or products of vs and us. Filter these monomials out
            valid_monom = False
            break

        if valid_monom:
          drake_basis.append(Monomial({drake_vars[idx]: powers[idx] for idx in var_idx}))
          sm_basis.append(np.prod([sm_vars[idx]**powers[idx] for idx in var_idx]))

    # Acceleration constraints: M*vdot - f = 0
    acc_constraints = np.zeros(self.nv, dtype=Expression)
    for i in range(self.nv):
      poly = sm.Poly(sm_constraints[i], sm_vars)
      for drake_monom, sm_monom in zip(drake_basis, sm_basis):
        coeff = poly.coeff_monomial(sm_monom)
        acc_constraints[i] += (coeff*drake_monom).ToExpression()

    # Trig constraints, s^2 + c^2 - 1 = 0
    trig_constraints = np.zeros(self.N, dtype=Expression)
    for i in range(self.N):
      trig_constraints[i] = q[1 + i]**2 + q[1 + self.N + i]**2 - 1

    return np.concatenate((trig_constraints, acc_constraints))

  def trig_lqr(self):
    # LQR in minimal coordinates
    A, B = self.lin_dynamics(np.zeros(self.nx_minimal), np.zeros(self.nu))
    Q = np.eye(self.nx_minimal)
    R = np.eye(self.nu)
    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.transpose()@S)

    # Convert to trigonometric coordinates
    Ktrig = np.zeros((self.nu, self.nq + self.nv))
    Ktrig[:, 0] = K[:, 0]
    Ktrig[:, -self.nv:] = K[:, -self.nv:]
    Strig = np.zeros((self.nq + self.nv, self.nq + self.nv))
    Strig[-self.nv:, -self.nv:] = S[-self.nv:, -self.nv:]
    Strig[0, 0] = S[0, 0]

    # Columns corresponding to theta in minimal coordinates correspond to sin(theta) in trig coordinates
    Ktrig[:, 1 + self.N:self.nq] = K[:, 1:self.nv]

    # (q_pris, sin)
    Strig[0, 1 + self.N:self.nq] = S[0, 1:self.nv]
    # (sin, q_pris)
    Strig[1 + self.N:self.nq, 0] = S[1:self.nv, 0]

    # (sin, sin)
    Strig[1 + self.N:self.nq, 1 + self.N:self.nq] = S[1:self.nv, 1:self.nv]

    # (sin, v)
    Strig[1 + self.N:self.nq, self.nq:self.nx] = S[1:self.nv, self.nq_minimal:self.nx_minimal]

    # (v, sin)
    Strig[self.nq:self.nx, 1 + self.N:self.nq] = S[self.nq_minimal:self.nx_minimal, 1:self.nv]

    return Strig, Ktrig, S, K
