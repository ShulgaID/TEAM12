import numpy as np
import math
from spatial.se import SE3, se3, skew
from spatial.so import SO3, so3



class Robot:
    def __init__(self):
        self.xfd = np.zeros(6)
        self.q = np.zeros(1)
        self.qd = np.zeros(1)
        self.model = {}
        self.positions = []

    def some_tree(self, nb, bf=1, skew=0, taper=1):
        model = {}
        model['gravity'] = np.array([0.0, -9.81, 0.0])    # WARN: hardcoded for now
        model['NB'] = nb
        
        self.q = np.zeros(nb)
        self.qd = np.zeros(nb)
        
        model['jtype'] = []
        model['parent'] = np.zeros(nb, dtype=int)
        model['Xtree'] = []
        model['I'] = []
        model['len'] = []

        
        for i in range(nb):
            model['jtype'].append('Rz')
            model['parent'][i] = int(np.floor((i + 1) - 2 + np.ceil(bf)) / bf)

            if model['parent'][i] == 0:
                model['Xtree'].append(
                    SE3(SO3().matrix, [0,0,0]).adjoint()
                )
            else:
                model['Xtree'].append(
                    SE3(SO3.rx(skew).T).adjoint() @ SE3(SO3().matrix, [model['parent'][i].size, 0, 0]).adjoint()
                )
            
            len_i = taper ** (i)
            mass = taper ** (3 * (i))
            CoM = len_i * np.array([0.5, 0, 0])
            Icm = mass * len_i**2 * np.diag([0.0025, 1.015/12, 1.015/12])

            model['I'].append(self.mcI(mass, CoM, Icm))
            model['len'].append(len_i)
        
        self.model = model

    def mcI(self, mass, com, Icm):
        I = np.zeros((6, 6))
        C = skew(com)
        I[0:3, 0:3] = Icm + mass * C @ C.T
        I[0:3, 3:6] = mass * C
        I[3:6, 0:3] = mass * C.T
        I[3:6, 3:6] = mass * np.eye(3)
        return I

if __name__=="__main__":
    robot = Robot()
    robot.some_tree(2, 1)

    np.set_printoptions(precision=4)
    for m in robot.model:
        print(f"{m}")
        if type(robot.model[m]) is list:
            for it in robot.model[m]:
                print(f"{it}")
        else:
            print(f"{robot.model[m]}")

