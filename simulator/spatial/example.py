import numpy as np

from so import SO3, so3
from se import SE3, se3, crm, crf

np.set_printoptions(precision=2, suppress=True)

# rx ry rz
v_rx = SO3.from_axis_angle((1,0,0), np.pi/2).matrix.T
v_ry = SO3.from_axis_angle((0,1,0), np.pi/2).matrix.T
v_rz = SO3.from_axis_angle((0,0,1), np.pi/2).matrix.T

# rot
q = np.pi/2
rotx = SE3(SO3.rx(q).T).adjoint()
roty = SE3(SO3.ry(q).T).adjoint()
rotz = SE3(SO3.rz(q).T).adjoint()

# print(rotx)

# xlt
xlt = SE3(SO3().matrix, [1,2,3]).adjoint()
print(xlt)


# plux plucker R,t -> X
R = SO3.from_axis_angle((1,0,0), np.pi/7).matrix.T
t = [1,2,3]
X = SE3(R, t).adjoint()

# plux plucker X -> R,t
(SE3.from_adjoint(X).matrix)

(crm([1,2,3,4,5,6]))
(crf([1,2,3,4,5,6]))
