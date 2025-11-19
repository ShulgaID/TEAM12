import numpy as np

class RNEAlgorithm:

    def __init__(self):
        pass

    def forward_dynamics(self, obj, q, qd, tau_ext, gravity):
        n = obj.n
        G = self._rnea(obj, q, np.zeros(n), np.zeros(n), gravity)
        C = self._rnea(obj, np.zeros(n), qd, np.zeros(n), np.zeros(3))

        M = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            M[:, i] = self._rnea(obj, q, np.zeros(n), e_i, np.zeros(3))

        try:
            # M qdd + C qd + G = tau_ext
            # qdd = int(M) (tau_ext - C -G)
            ddq = np.linalg.solve(M, tau_ext - G  - C)
        except np.linalg.LinAlgError as e:
            logger.error("Mass matrix is singular: %s", e)
            raise

        return ddq

    def _rotz(self, q):
        return np.array([
                    [np.cos(q), -np.sin(q), 0],
                    [np.sin(q), np.cos(q), 0],
                    [0,0,1]
        ])

    def _rnea(self, obj, q, qd, qdd, gravity):
        """
        tau = ID(q, dq, ddq, gravity)
        """
        n = obj.n
        tau = np.zeros(n)

        omega = [np.zeros(3)] * (n + 1)
        omega_dot = [np.zeros(3)] * (n + 1)
        v_dot = [np.zeros(3)] * (n + 1)
        v_dot_c = [np.zeros(3)] * (n + 1)
        forces = [np.zeros(3) for i in range(n + 2)]
        moments = [np.zeros(3) for i in range(n + 2)]

        v_dot[0] = -gravity

        z_axis = np.array([0, 0, 1])
        
        for i in range(1, n + 1):
            link = obj.links[i - 1]
            is_revolute = link.joint_type == 'revolute'
            joint_axis = z_axis if is_revolute else link.z_axis

            E = self._rotz(q[i - 1]).T if is_revolute else np.eye(3)
            r_i = np.array([link.length, 0.0, 0.0]) # vector from joint i to i + 1
            c_i = link.com

            if is_revolute:
                omega[i] = E @ (omega[i - 1] + qd[i - 1] * z_axis)
                omega_dot[i] = E @ (
                    omega_dot[i - 1] + 
                    qdd[i - 1] * z_axis +  # sadsalkdldkjas
                    np.cross(omega[i], qd[i - 1] * z_axis)
                )
            else:
                omega[i] = E @ omega[i - 1]
                omega_dot[i] = E @ omega_dot[i - 1]

            v_dot[i] = (
                E @ v_dot[i - 1] + 
                np.cross(omega_dot[i], r_i) + 
                np.cross(omega[i], np.cross(omega[i], r_i))
                 + (qdd[i - 1] * joint_axis if not is_revolute else np.zeros(3))
            )
            v_dot_c[i] = (
                v_dot[i] + 
                np.cross(omega_dot[i], c_i) + 
                np.cross(omega[i], np.cross(omega[i], c_i))
            )

        for i in reversed(range(1, n+1)):
            link = obj.links[i - 1]

            is_revolute = link.joint_type == 'revolute'
            E = self._rotz(q[i + 1]).T if i < n - 1 else np.eye(3)
            joint_axis = z_axis if is_revolute else link.z_axis

            r_i = np.array([link.length, 0.0, 0.0])
            c_i = link.com

            F_i = link.mass * v_dot_c[i]
            N_i = link.inertia @ omega_dot[i] + np.cross(
                omega[i], link.inertia @ omega[i]
            )

            forces[i] = F_i + E @ forces[i + 1]
            moments[i] = (
                N_i 
                + E @ moments[i + 1] 
                + np.cross(E @ forces[i + 1], c_i) 
                - np.cross(forces[i], r_i + c_i)
            )

            tau[i - 1] = joint_axis @ (moments[i] if is_revolute else forces[i])

        return tau