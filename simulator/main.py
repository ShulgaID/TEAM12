import numpy as np

from physics import PhysicsEngine
from renderer import Renderer
from world import World

from dynamics import RNEAlgorithm
from controller import PIDController, CartPoleLQRController


from objects import Link, RobotTwoLink

def robot_ext_pid_controlled(dynamics_algo, links):
    for i in range(1):
        l = Link(mass=1.0, inertia=np.diag([0.1,0.1,0.5]), length=1.0, com=np.array([-0.5, 0, 0]), joint_type='revolute')
        links.append(l)

    q_init = np.array([0.0] * len(links))
    qd_init = np.array([0.0] * len(links))

    kp = []
    kd = []
    kp.extend([50.0] * (len(links)))
    kd.extend([25.0] * (len(links)))

    pid_controler = PIDController(
        kp = np.array(kp),
        ki = np.array([0.0] * len(links)),
        kd = np.array(kd),
        target_state = np.array([2.0, 0.5]) # in range [-0.5, 0.5]
        # target_state = np.array(np.random.random(len(links)) - 0.5) # in range [-0.5, 0.5]
    )

    twolink = RobotTwoLink(links, q_init, qd_init, dynamics_algo, pid_controler)
    return twolink

def robot_cartpole(dynamics_algo, links):
    q_init = np.array([0.0, np.pi / 2 + np.pi / 6])
    qd_init = np.array([0.0, 0.0])
    cp_lqr_controler = CartPoleLQRController(
        links=links,
        gravity=9.81,
        target_state = np.array([0.0, np.pi / 2, 0.0, 0.0])
    )
    twolink = RobotTwoLink(links, q_init, qd_init, dynamics_algo, cp_lqr_controler)
    return twolink

def main():

    physics = PhysicsEngine(gravity=[0.0, -9.81, 0.0])
    renderer = Renderer()

    world = World(physics, renderer)

    dynamics_algo = RNEAlgorithm()

    links = [
        Link(mass=1.0, inertia=np.diag([0.1,0.1,0.5]), length=1.0, com=np.array([-0.5, 0, 0]), joint_type='prismatic'),
        Link(mass=0.1, inertia=np.diag([0.01,0.01,0.05]), length=1.0, com=np.array([-0.5, 0, 0]), joint_type='revolute'),
    ]

    ## there are two examples here

    ## 1. example
    # robot = robot_ext_pid_controlled(dynamics_algo, links)

    # 2. example
    robot = robot_cartpole(dynamics_algo, links)

    world.add_object(robot)
    world.run(5000)

if __name__=="__main__":
    main()