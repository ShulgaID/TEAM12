from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle  # noqa: F401


class Renderer:
    """2D renderer for robot simulation.

    Args:
        x_limits: Horizontal axis range.
        y_limits: Vertical axis range.
        max_colors: Number of colours in the rainbow palette (one per link).
        record: If True, capture every rendered frame for later export.
    """

    def __init__(
        self,
        x_limits: tuple[float, float] = (-10.0, 10.0),
        y_limits: tuple[float, float] = (-10.0, 10.0),
        max_colors: int = 20,
        record: bool = False,
    ) -> None:
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)
        self.ax.set_aspect("equal")

        # World-frame axes arrows
        self.ax.annotate(
            "",
            xy=(x_limits[1] / 10, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.5),
        )
        self.ax.annotate(
            "",
            xy=(0, y_limits[1] / 10),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="green", alpha=0.5),
        )

        plt.show(block=False)

        self.links_lines = None
        self.joints_circles = None
        self.colors = plt.cm.rainbow(np.linspace(0, 1, max_colors))
        self.base_patch = None

        # Recording
        self.record = record
        self._frames: list[np.ndarray] = []

    def update(self, objects, dt: float = 0.0001) -> None:
        """Redraw all objects and optionally capture a frame."""
        if not objects:
            return

        def draw_tree(obj, q):
            parents = obj.model["parent"]
            nodes = []
            edges = []
            angles = [0.0] * len(parents)
            length = 1.0

            for i in range(len(parents)):
                parent = parents[i]
                if parent == -1:
                    x_p, y_p = 0.0, 0.0
                    angles[i] = q[i]
                else:
                    x_p, y_p = nodes[parent]
                    angles[i] = angles[parent] + q[i]

                x_child = x_p + length * np.cos(angles[i])
                y_child = y_p + length * np.sin(angles[i])
                nodes.append((x_child, y_child))
                edges.append((x_p, y_p, x_child, y_child))

            return nodes, edges

        all_links = []
        all_points = []

        for obj in objects:
            nodes, edges = draw_tree(obj, obj.q)
            seg = np.array([[[xp, yp], [xc, yc]] for xp, yp, xc, yc in edges])
            pts = np.array(nodes)
            all_links.append(seg)
            all_points.append(pts)

        merged_links = np.concatenate(all_links, axis=0)
        merged_points = np.concatenate(all_points, axis=0)

        n_links = len(merged_links)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_links))

        if self.links_lines is None:
            self.links_lines = LineCollection(merged_links, colors=colors, linewidths=3)
            self.ax.add_collection(self.links_lines)
            self.joints_circles = self.ax.scatter(
                merged_points[:, 0], merged_points[:, 1],
                c="lightblue", s=40, zorder=10,
            )
            self.ax.scatter(0.0, 0.0, c="blue", s=50, zorder=10)
        else:
            self.links_lines.set_segments(merged_links)
            self.links_lines.set_colors(colors)
            self.joints_circles.set_offsets(merged_points)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        if self.record:
            self._capture_frame()

    def _capture_frame(self) -> None:
        """Grab the current figure as an RGBA numpy array."""
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        w, h = self.fig.canvas.get_width_height()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4).copy()
        self._frames.append(frame)

    def save(self, path: str | os.PathLike, fps: int = 30) -> None:
        """Save recorded frames to *path* (.mp4 or .gif).

        Args:
            path: Output file path; extension determines format.
            fps:  Frames per second.

        Raises:
            RuntimeError: If no frames have been captured.
            ValueError:   If the file extension is not supported.
        """
        if not self._frames:
            raise RuntimeError(
                "No frames recorded. Create the renderer with record=True "
                "and run the simulation before calling save()."
            )

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".mp4":
            self._save_mp4(path, fps)
        elif suffix == ".gif":
            self._save_gif(path, fps)
        else:
            raise ValueError(f"Unsupported format '{suffix}'. Use '.mp4' or '.gif'.")

    def _save_mp4(self, path: Path, fps: int) -> None:
        import imageio
        rgb_frames = [f[:, :, :3] for f in self._frames]
        imageio.mimwrite(str(path), rgb_frames, fps=fps, format="FFMPEG", codec="libx264")
        print(f"[Renderer] Saved MP4 ({len(rgb_frames)} frames, {fps} fps) -> {path}")

    def _save_gif(self, path: Path, fps: int) -> None:
        import imageio
        duration = 1000 // fps
        rgb_frames = [f[:, :, :3] for f in self._frames]
        imageio.mimwrite(str(path), rgb_frames, duration=duration, loop=0)
        print(f"[Renderer] Saved GIF ({len(rgb_frames)} frames, {fps} fps) -> {path}")

    def close(self) -> None:
        plt.close(self.fig)
