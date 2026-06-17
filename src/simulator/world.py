from __future__ import annotations

import time
from pathlib import Path


class World:
    def __init__(self, physics, renderer):
        self.physics = physics
        self.renderer = renderer
        self.objects = []
        self.time = 0.0
        self.dt = 0.02
        self.plane = None  # optional ground plane

    def set_plane(self, plane):
        self.plane = plane

    def add_object(self, obj):
        self.objects.append(obj)

    def step(self, i: int) -> None:
        self.physics.update(self.objects, self.dt)
        if i % 2 == 0:
            self.renderer.update(self.objects)
        self.time += self.dt

    def run(self, steps: int, save_path: str | None = None, fps: int = 30) -> None:
        """Run the simulation for *steps* physics steps.

        Args:
            steps:     Number of physics integration steps.
            save_path: If provided (e.g. 'out.mp4' or 'out.gif'),
                       saves the recorded frames after the loop.
                       Recording must be enabled on the renderer
                       (Renderer(record=True)).
            fps:       Frames-per-second for the saved video/GIF.
        """
        for i in range(steps):
            self.step(i)
            time.sleep(self.dt)

        if save_path is not None:
            self.renderer.save(save_path, fps=fps)
