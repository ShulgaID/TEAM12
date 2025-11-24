import numpy as np

from physics import PhysicsEngine
from renderer import Renderer
from world import World

from objects import Robot
from dynamics.ab_algorithm import ABAlgorithm


def main():

    fd_solver = ABAlgorithm()

    physics = PhysicsEngine(fd_solver, gravity=[0.0, -9.81, 0.0])
    renderer = Renderer()

    world = World(physics, renderer)

    robot = Robot()
    robot.some_tree(2)

    world.add_object(robot)
    world.run(5000)

    # try:
    #     world.run(5000)
    # except Exception as e:
    #     print(e)

if __name__=="__main__":
    main()