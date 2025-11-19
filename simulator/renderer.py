import logging
from typing import List, Optional, Protocol, Any

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Renderer:

    def __init__(
        self,
        x_limits: tuple[float, float] = (-20.0, 20.0),
        y_limits: tuple[float, float] = (-5.0, 5.0),
        max_colors: int = 20
    ) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)
        self.ax.set_aspect('equal')

        plt.show(block=False)

        self.links_lines = None
        self.joints_circles = None
        self.colors = plt.cm.rainbow(np.linspace(0, 1, max_colors))

        logger.debug("Renderer x=%s, y=%s", x_limits, y_limits)

    def update(self, objects, dt=0.0001):

            if not objects:
                logger.debug("No objects to render")
                return

            all_links = []
            all_points = []
            
            for obj in objects:
                obj.positions = obj.forward_kinematics(obj.q)

                xs = obj.positions[:, 0]
                ys = obj.positions[:, 1]
                
                links = np.column_stack([xs[:-1], ys[:-1], xs[1:], ys[1:]]).reshape(-1, 2, 2)
                all_links.append(links)
                
                all_points.append(np.column_stack([xs[:-1], ys[:-1]]))
            
            all_links = np.concatenate(all_links) if all_links else np.empty((0, 2, 2))
            all_points = np.concatenate(all_points) if all_points else np.empty((0, 2))
            
            if self.links_lines is None:
                self.links_lines = LineCollection(all_links, colors=self.colors, 
                                        linewidths=2, capstyle='round')
                self.ax.add_collection(self.links_lines)

                self.joints_circles = self.ax.scatter(all_points[:, 0], all_points[:, 1], 
                                                c=self.colors[:len(all_points)], 
                                                s=25, zorder=10)
            else:
                self.links_lines.set_segments(all_links)
                self.joints_circles.set_offsets(all_points)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def close(self) -> None:
        plt.close(self.fig)
        logger.debug("Render window closed")