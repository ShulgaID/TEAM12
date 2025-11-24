import numpy as np

class PhysicsEngine:

    def __init__(self, fd_solver, gravity=[0.0, -9.81, 0.0]):
        self.fd_solver = fd_solver
        self.gravity = np.array(gravity)

    def update(self, objects, dt):
        if not objects:
            return

        for obj in objects:
            
            tau = np.zeros(obj.qd.shape)
            # tau = -0.1 * obj.qd
            # tau = 100.0 * (1.57 * np.ones_like(obj.q) - obj.q) - 2.0 * obj.qd


            ddq = self.fd_solver.forward_dynamics(obj, obj.q, obj.qd, tau)
            obj.qd += ddq * dt
            obj.q += obj.qd * dt
