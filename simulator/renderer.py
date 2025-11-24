
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection

class Renderer:
    
    def __init__(
        self,
        x_limits = (-10.0, 10.0),
        y_limits = (-10.0, 10.0),
        max_colors = 20
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
        
        self.base_patch = None

    def update(self, objects, dt=0.0001):

            if not objects:
                return

            all_links = []
            all_points = []

            def draw_tree(parents, q):
                nodes = [(0.0, 0.0)]
                edges = []
                angles = [0.0] * len(parents)
                
                for i in range(0, len(parents)):
                    parent = parents[i]

                    x_p, y_p = nodes[parent]
                    angles[i] = angles[parent] + q[i]

                    length = 1.0
                    
                    x_child = x_p + length * np.cos(angles[i])
                    y_child = y_p + length * np.sin(angles[i])
                    
                    nodes.append((x_child, y_child))
                    edges.append((x_p, y_p, x_child, y_child))
                
                return nodes, edges

            for obj in objects:
                
                nodes, edges = draw_tree(obj.model['parent'], obj.q)

                links = []
                for i in range(1, len(obj.q)):
                    parent = obj.model['parent'][i]
                    x_p, y_p = nodes[parent]
                    x_c, y_c = nodes[i]
                    links.append([[x_p, y_p], [x_c, y_c]])

                links = np.array(links)
                points = np.array(nodes)
                
                if self.links_lines is None:
                    self.links_lines = LineCollection(links, colors=self.colors, linewidths=2)
                    self.ax.add_collection(self.links_lines)
                    
                    self.joints_circles = self.ax.scatter(
                        points[:, 0], points[:, 1], 
                        c='lightblue', s=20, zorder=10
                    )
                else:
                    self.links_lines.set_segments(links)
                    self.joints_circles.set_offsets(points)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)
        logger.debug("Render window closed")
