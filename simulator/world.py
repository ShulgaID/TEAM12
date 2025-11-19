
class World:
    def __init__(self, physics, renderer):
        self.physics = physics
        self.renderer = renderer
        self.plane = None
        self.objects = []
        self.time = 0
        self.dt = 0.01

    def set_plane(self, plane):
        self.plane = plane

    def add_object(self, obj):
        self.objects.append(obj)

    def step(self, i):
            self.physics.update(self.objects, self.plane, self.dt)
            if i % 10 == 0:
                self.renderer.update(self.objects, self.plane)
            self.time += self.dt

    def run(self, steps):
        for i in range(steps):
            self.step(i)
            # time.sleep(0.01)