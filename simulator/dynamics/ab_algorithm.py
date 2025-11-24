import numpy as np

from spatial.so import SO3, so3
from spatial.se import SE3, se3, crm, crf


class ABAlgorithm:

    def __init__(self):
        pass

    def forward_dynamics(self, obj, q, qd, tau, f_ext=None):
        model = obj.model

        a_grav = self.get_gravity(model)
        NB = model['NB']

        (Xup, S, v, c, IA, pA, U, d, u, a) = ([None] * NB for _ in range(10))
        qdd = np.zeros(NB)

        for i in range(NB):
            print(model['parent'][i], i)

            XJ, S[i] = self.jcalc(model['jtype'][i], q[i])
            vJ = S[i] * qd[i]
            Xup[i] = XJ @ model['Xtree'][i]
            if model['parent'][i] == -1:
                v[i] = vJ
                c[i] = np.zeros_like(a_grav)
            else:
                print(v[model['parent'][i]])
                v[i] = Xup[i] @ v[model['parent'][i]] + vJ
                c[i] = crm(v[i]) @ vJ
            IA[i] = model['I'][i]
            pA[i] = crf(v[i]) @ model['I'][i] @ v[i]
        
        if f_ext is not None:
            pA = self.apply_external_forces(model['parent'], Xup, pA, f_ext)
        
        for i in range(NB-1, -1, -1):
            U[i] = IA[i] @ S[i]
            d[i] = S[i].T @ U[i]
            u[i] = tau[i] - S[i].T @ pA[i]
            if model['parent'][i] != 0:
                Ia = IA[i] - U[i] / d[i] @ U[i].T
                pa = pA[i] + Ia @ c[i] + U[i] * u[i] / d[i]
                IA[model['parent'][i]] = IA[model['parent'][i]] + Xup[i].T @ Ia @ Xup[i]
                pA[model['parent'][i]] = pA[model['parent'][i]] + Xup[i].T @ pa
        
        for i in range(NB):
            if model['parent'][i] == 0:
                a[i] = Xup[i] @ -a_grav + c[i]
            else:
                a[i] = Xup[i] @ a[model['parent'][i]] + c[i]
            qdd[i] = ((u[i] - U[i].T @ a[i]) / d[i])[-1][-1]
            a[i] = a[i] + S[i] * qdd[i]
    
        return qdd

    def get_gravity(self, model):
        g = np.array(model['gravity'])
        a_grav = np.array([0, 0, 0, g[0], g[1], g[2]])
        return a_grav

    def apply_external_forces(self, parent, Xup, pA, f_ext):
        pass

    def jcalc(self, jtyp, q):
        joint_map = {
            'Rx': (lambda q: SE3(SO3.rx(q).T).adjoint(), np.array([[1], [0], [0], [0], [0], [0]])),
            'Ry': (lambda q: SE3(SO3.ry(q).T).adjoint(), np.array([[0], [1], [0], [0], [0], [0]])),
            'Rz': (lambda q: SE3(SO3.rz(q).T).adjoint(), np.array([[0], [0], [1], [0], [0], [0]])),
            'Px': (lambda q: SE3(SO3().matrix, [q, 0, 0]).adjoint(), np.array([[0], [0], [0], [1], [0], [0]])),
            'Py': (lambda q: SE3(SO3().matrix, [0, q, 0]).adjoint(), np.array([[0], [0], [0], [0], [1], [0]])),
            'Pz': (lambda q: SE3(SO3().matrix, [0, 0, q]).adjoint(), np.array([[0], [0], [0], [0], [0], [1]])),
        }

        code = jtyp if isinstance(jtyp, str) else jtyp.get('code')

        transform_fn, S = joint_map[code]

        Xj = transform_fn(q)

        return Xj, S        
