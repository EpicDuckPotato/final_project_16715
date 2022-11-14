import sympy.physics.mechanics as me
import sympy as sm
import numpy as np

class NLinkCartpole(object):
  def __init__(self, N, link_length, link_mass):
    self.N = N
    self.nx = 2 + 2*N
    self.nu = N

    g = 9.81
    t = sm.symbols('t')

    self.q = me.dynamicsymbols('q:{}'.format(N + 1)) # Generalized coordinates
    self.qdot = me.dynamicsymbols('qdot:{}'.format(N + 1)) # Generalized velocities
    self.u = me.dynamicsymbols('u:{}'.format(N)) # Input forces (none for the last joint)

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
    return self.nx, self.nu

  # u actuates the cart, and all revolute joints except the last (top) one
  def dynamics(self, x, u):
    xdot = np.zeros(self.nx)
    xdot[:self.nx//2] = np.copy(x[self.nx//2:])
    xu = np.concatenate((x, u))
    xdot[self.nx//2:] = np.linalg.solve(self.mass_matrix_fn(*xu), self.forcing_fn(*xu)).flatten()
    return xdot

  def lin_dynamics(self, x, u):
    equilibrium_dict = dict(zip(self.q + self.qdot, x))
    F_A, F_B, r = self.kane.linearize(new_method=True, op_point=equilibrium_dict, A_and_B=True)
    A = sm.matrix2numpy(F_A, dtype=float)
    B = sm.matrix2numpy(F_B, dtype=float)
    return A, B
