import time


class World:
    def __init__(self, physics, renderer):
        self.physics = physics
        self.renderer = renderer
        self.objects = []
        self.time = 0
        self.dt = 0.01

    def set_plane(self, plane):
        self.plane = plane

    def add_object(self, obj):
        self.objects.append(obj)

    def step(self, i):
        self.physics.update(self.objects, self.dt)
        if i % 5 == 0:
            self.renderer.update(self.objects)
        self.time += self.dt

    def run(self, steps):
        for i in range(steps):
            self.step(i)
            time.sleep(self.dt)
