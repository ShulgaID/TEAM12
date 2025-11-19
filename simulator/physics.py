import numpy as np

class PhysicsEngine:

    def __init__(self, gravity=[0.0, -9.81, 0.0]):
        self.gravity = np.array(gravity)

    def update(self, robots, plane, dt):
        for robot in robots:

            tau_u = robot.compute_control(dt)
            
            qdd = robot.forward_dynamics(tau_u, self.gravity)

            robot.qd += qdd * dt
            robot.q += robot.qd * dt

            robot.positions = robot.forward_kinematics()