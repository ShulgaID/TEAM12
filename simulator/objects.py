import numpy as np

class Ball:
    def __init__(self, x=0.0, y=0.0, mass=1.0, radius=0.2, v_x=0.0, v_y=0.0, omega = 0.0, color = "red"):
        self.color = color
        self.radius = radius
        self.mass = mass
        self.inertia = 0.4 * self.mass * self.radius**2

        # state
        self.pos = np.array([x, y])
        self.v = np.array([v_x, v_y])
        self.omega = omega

class Plane:
    def __init__(self, y=0.0, angle=0.0):
        # y = kx + b
        self.y = y
        self.angle = angle


class Link:
    def __init__(self, mass=1.0, inertia=np.diag([0.1, 0.1, 0.1]), length=1.0, com=[0.5, 0.0, 0.5], joint_type='revolute', z_axis=[1,0,0]):
        self.mass = mass
        self.inertia = inertia
        self.length = length
        self.com = com

        self.joint_type = joint_type
        self.z_axis = np.array(z_axis)

    def r_i_ci(self):
        # rc
        return self.com

    def r_i_minus_1_i(self):
        # r
        return np.array([self.length, 0, 0])
    
    def r_i_i_plus_1(self):
        # r2
        return np.array([self.length, 0, 0])


class RobotTwoLink:
    def __init__(self, links, q_init, qd_init, fd_solver, controller):
        self.links = links
        self.n = len(self.links)
        self.q = np.array(q_init)
        self.qd = np.array(qd_init)
        
        self.fd_solver = fd_solver
        self.controller = controller

        self.positions = self.forward_kinematics(self.q)

    def compute_control(self, dt, target_state = None):
        return self.controller.compute_control(self, dt, target_state)

    def forward_dynamics(self, tau, gravity):
        return self.fd_solver.forward_dynamics(self, self.q, self.qd, tau, gravity)

    def forward_kinematics(self, q = None):
        if q is None:
            q = self.q

        positions = [np.array([0.0, 0.0])]
        theta = 0
        for i, link in enumerate(self.links):
            if link.joint_type == 'revolute':
                theta += q[i]
                r = link.length * np.array([np.cos(theta), np.sin(theta)])
            else:
                r = q[i] * link.z_axis[:2]
            positions.append(positions[-1] + r)
        return np.array(positions)
