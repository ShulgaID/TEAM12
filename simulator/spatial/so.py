import numpy as np

class so3:

    def __init__(self, w):
        self.w = np.array(w, dtype=float).reshape(3)

    @property
    def matrix(self):
        return so3.hat(self.w)

    @staticmethod
    def hat(w):
        # R^3 → so(3) 
        w = np.array(w).reshape(3)
        return np.array([
            [0,     -w[2],  w[1]],
            [w[2],   0,    -w[0]],
            [-w[1],  w[0],  0]
        ])

    @staticmethod
    def vee(W):
        # so(3) → R^3
        return np.array([W[2,1], W[0,2], W[1,0]])

    def exp(self):
        theta = np.linalg.norm(self.w)
        if theta < 1e-9:
            return SO3(np.eye(3))  # near zero → identity
        axis = self.w / theta
        K = so3.hat(axis)
        R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K
        return SO3(R)

    def __add__(self, other):
        return so3(self.w + other.w)


class SO3:

    def __init__(self, R=None):
        self._R = np.eye(3) if R is None else np.array(R, dtype=float)

    @property
    def matrix(self):
        return self._R

    @staticmethod
    def identity():
        return SO3()

    def compose(self, other):
        return SO3(self._R @ other.matrix)

    def inverse(self):
        return SO3(self._R.T)

    def adjoint(self):
        return self._R

    def act(self, p):
        return self._R @ np.array(p).flatten()

    @staticmethod
    def from_axis_angle(axis, theta):
        axis = np.array(axis).reshape(3)
        axis = axis / np.linalg.norm(axis)
        return so3(axis * theta).exp()

    @staticmethod
    def from_quaternion(q):
        """q = [x, y, z, w]"""
        q = np.array(q, dtype=float).flatten()
        q = q / np.linalg.norm(q)
        w, x, y, z = q

        q0s = w * w
        q1s = x * x
        q2s = y * y
        q3s = z * z

        q01 = w * x
        q02 = w * y
        q03 = w * z
        q12 = x * y
        q13 = x * z
        q23 = y * z

        R = 2 * np.array([
            [q0s + q1s - 0.5, q12 + q03,     q13 - q02],
            [q12 - q03,       q0s + q2s - 0.5, q23 + q01],
            [q13 + q02,       q23 - q01,     q0s + q3s - 0.5]
        ])

        return SO3(R)

    @staticmethod
    def rx(theta):
        return SO3.from_axis_angle((1,0,0), theta).matrix

    @staticmethod
    def ry(theta):
        return SO3.from_axis_angle((0,1,0), theta).matrix

    @staticmethod
    def rz(theta):
        return SO3.from_axis_angle((0,0,1), theta).matrix
