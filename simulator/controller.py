from abc import ABC, abstractmethod
from typing import Optional, Union, Any, List
import numpy as np
from scipy.linalg import solve_continuous_are
from numpy.typing import NDArray

class PIDController:
    def __init__(self, kp, ki, kd, target_state=None):
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        
        self.target_state = (
            np.array(target_state, dtype=float) if target_state is not None else np.zeros_like(kp)
        )
        
        self.integral = np.zeros_like(kp)
        self.prev_error = np.zeros_like(kp)
        self.integral_limit = 100.0

    def compute_control(self, system, dt, target_state=None):
        current_state = system.q
        error = self.target_state - current_state
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        return control


class CartPoleLQRController:
    def __init__(self, links, gravity, target_state, Q=None, R=None):
        if len(links) != 2:
            raise ValueError("Cart pole must have 2 links")
        
        self.mass1 = links[0].mass
        self.mass2 = links[1].mass
        self.length = links[1].length
        com = links[1].com
        self.l = abs(com[0]) if com[0] != 0 else self.length
        self.g = gravity
        
        self.Q = Q if Q is not None else np.diag([0.01, 100.0, 0.1, 10.0])
        self.R = R if R is not None else np.array([[10.0]])         

        self.A, self.B = self._compute_model()
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ P
        
        self.target_state = target_state
        self.E_target = self.mass2 * self.g * self.l

    def _compute_model(self):
        m1, m2, l, g = self.mass1, self.mass2, self.l, self.g
        
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, m2*g/m1, 0, 0],
            [0, (m1+m2)*g/(m1*l), 0, 0]
        ])
        
        B = np.array([
            [0],
            [0], 
            [1/m1],
            [1/(m1*l)]
        ])
        
        return A, B

    def _compute_energy(self, theta, theta_dot):
        T = 0.5 * self.mass2 * (self.l * theta_dot) ** 2
        V = self.mass2 * self.g * self.l * (1 - np.cos(theta - np.pi / 2))
        return T + V

    def compute_control(self, obj, dt, target_state=None):
        if target_state is not None:
            self.target_state = np.array([target_state[0], np.pi / 2, 0, 0])
        
        current = np.array([obj.q[0], obj.q[1], obj.qd[0], obj.qd[1]])
        
        theta_error = current[1] - np.pi / 2
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
        current[1] = np.pi / 2 + theta_error
        
        E = self._compute_energy(current[1], current[3])
        
        if abs(theta_error) < 1.0:  # LQR stabilization
            error = current - self.target_state
            force = -float(self.K @ error)  
        else:  # Energy-based swing-up
            theta = current[1]
            theta_dot = current[3]
            E_error = E - self.E_target
            force = -10.0 * E_error * np.sign(theta_dot * np.cos(theta))

        force = np.clip(force, -500, 500)
        # return np.array([force, 0.0]) - np.array([[2.0, 0.0],[0.0, 0.1]]) @ np.array([current[2], current[3]])  # add friction to more stability
        return np.array([force, 0.0])