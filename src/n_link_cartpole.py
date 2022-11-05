import pinocchio as pin
import numpy as np

class NLinkCartpole(object):
  def __init__(self, N, link_length, link_mass):
    self.N = N
    self.pin_model = pin.Model()
    # Add the cart joint (prismatic joint along x axis)
    jidx = 0
    joint_model = pin.JointModelPrismaticUnaligned()
    joint_model.axis[0] = 1
    joint_model.axis[1] = 0
    joint_model.axis[2] = 0
    jidx = self.pin_model.addJoint(jidx, joint_model, pin.SE3.Identity(), 'cart_joint')

    # Add cart link
    inertia = pin.Inertia.FromBox(link_mass, link_length, link_length, link_length)
    link_se3 = pin.SE3.Identity()
    self.pin_model.appendBodyToJoint(jidx, inertia, link_se3)
    self.pin_model.addBodyFrame('cart_link', jidx, link_se3, -1)

    # Add the additional revolute joints
    for i in range(N):
      joint_se3 = pin.SE3.Identity()

      if i != 0:
        joint_se3.translation[2] = link_length

      if i == 0:
        # The joint rotates about its x axis. Align the joint's x axis with the global y axis
        joint_se3.rotation = pin.exp3(np.array([0, 0, -np.pi/2]))

      joint_model = pin.JointModelRevoluteUnaligned()
      joint_model.axis[0] = 1
      joint_model.axis[1] = 0
      joint_model.axis[2] = 0
      jidx = self.pin_model.addJoint(jidx, joint_model, joint_se3, 'cart_joint')

      # Add link
      inertia = pin.Inertia.FromCylinder(link_mass, 0.05, link_length)
      link_se3 = pin.SE3.Identity()
      link_se3.translation[2] = link_length/2
      self.pin_model.appendBodyToJoint(jidx, inertia, link_se3)
      self.pin_model.addBodyFrame('pole_link' + str(i), jidx, link_se3, -1)

    self.pin_data = pin.Data(self.pin_model)
    self.nq = self.pin_model.nq
    self.nv = self.pin_model.nv
    self.nx = self.nq + self.nv
    self.nu = self.pin_model.nv - 1

  def get_dim(self):
    return self.nx, self.nu

  def unpack(self, x, u):
    q = x[:self.nq]
    v = x[self.nq:]
    tau = np.zeros(self.nv)
    tau[:-1] = u
    return q, v, tau

  # u actuates the cart, and all revolute joints except the last (top) one
  def dynamics(self, x, u):
    q, v, tau = self.unpack(x, u)

    xdot = np.zeros(self.nq + self.nv)
    xdot[:self.nq] = np.copy(v)
    xdot[self.nv:] = pin.aba(self.pin_model, self.pin_data, q, v, tau)

    return xdot

  def lin_dynamics(self, x, u):
    q, v, tau = self.unpack(x, u)

    A = np.zeros((2*self.nv, 2*self.nv))
    B = np.zeros((2*self.nv, self.nu))

    dtau_du = np.zeros((self.nv, self.nu))
    dtau_du[:-1] = np.eye(self.nu)

    A[:self.nv, self.nv:] = np.eye(self.nv)

    pin.computeABADerivatives(self.pin_model, self.pin_data, q, v, tau)
    A[self.nv:, :self.nv] = self.pin_data.ddq_dq
    A[self.nv:, self.nv:] = self.pin_data.ddq_dv
    B[self.nv:] = self.pin_data.Minv@dtau_du

    return A, B
