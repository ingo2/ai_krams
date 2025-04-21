"""
Utility to convert a 2D image into an (x, y, value) point cloud.

- x and y are mapped to [-1, 1] with the image centre as origin while the
  aspect ratio of the input is preserved.
- value is a float in the range 0..255. By default it is the mean of the
  R, G and B channels. Pass custom weights (wr, wg, wb) to use a different
  grayscale conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

Weights = Tuple[float, float, float]


def _validate_weights(weights: Weights) -> np.ndarray:
    """Return the weights as a normalised NumPy array of shape (3,)."""
    if len(weights) != 3:
        raise ValueError("weights must be a 3‑tuple (wr, wg, wb)")
    w = np.asarray(weights, dtype=np.float32)
    if np.any(w < 0):
        raise ValueError("weights must be non‑negative")
    s = float(w.sum())
    if s == 0:
        raise ValueError("weights must not all be zero")
    return w / s


class PointCloudifier:
    """Create and handle point clouds derived from raster images."""

    def __init__(self) -> None:
        # List of points in the format (x, y, value).
        self.points: List[Tuple[float, float, float]] = []

    def cloudify_image(
        self,
        filename: str | Path,
        *,
        sample_rate: int = 1,
        weights: Optional[Weights] = None,
    ) -> List[Tuple[float, float, float]]:
        """Convert *filename* into a point cloud.

        Parameters
        ----------
        filename : str or Path
            Path to the image (any Pillow‑compatible format).
        sample_rate : int, default 1
            Grid step size (1 means every pixel, 2 every second pixel, etc.).
        weights : tuple or None
            Channel weights (wr, wg, wb). The tuple is normalised internally.

        Returns
        -------
        list of (float, float, float)
            [(x, y, value), ...] with x and y in [-1, 1] and value in
            [0, 255] (float).
        """
        if sample_rate < 1:
            raise ValueError("sample_rate must be an integer >= 1")

        img = Image.open(filename).convert("RGB")
        w, h = img.size
        arr = np.asarray(img, dtype=np.uint8)

        # Build the regular sampling grid.
        xs = np.arange(0, w, sample_rate)
        ys = np.arange(0, h, sample_rate)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        coords_x = grid_x.ravel()
        coords_y = grid_y.ravel()
        pixels = arr[coords_y, coords_x]

        # Calculate grayscale intensity per sampled pixel.
        if weights is None:
            values = pixels.mean(axis=1, dtype=np.float32)
        else:
            w_arr = _validate_weights(weights)
            values = (pixels * w_arr).sum(axis=1, dtype=np.float32)
        values = np.clip(values, 0.0, 255.0)

        # Normalize pixel coordinates to the range [-1, 1].
        max_dim = max(w, h)
        norm_x = 2.0 * coords_x / (max_dim - 1) - w / max_dim
        norm_y = -(2.0 * coords_y / (max_dim - 1) - h / max_dim)

        self.points = list(zip(norm_x.tolist(), norm_y.tolist(), values.tolist()))
        return self.points

    def save_json(self, filepath: str | Path, *, indent: int = 2) -> None:
        """Save the current point cloud to *filepath* in JSON format."""
        if not self.points:
            raise RuntimeError("No point cloud to save. Run cloudify_image() first.")
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.points, fh, indent=indent)

    def load_json(self, filepath: str | Path) -> List[Tuple[float, float, float]]:
        """Load a point cloud that was previously created with save_json()."""
        with open(filepath, "r", encoding="utf-8") as fh:
            self.points = json.load(fh)
        return self.points

    def plot(
        self,
        *,
        cmap: str = "plasma",
        point_size: float = 1.0,
        title: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Display the current point cloud with Matplotlib."""
        if not self.points:
            raise RuntimeError("No point cloud to plot. Generate or load one first.")

        xs, ys, vals = zip(*self.points)
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(xs, ys, c=vals, s=point_size, cmap=cmap, marker=".")
        plt.axis("equal")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel("x (normalized)")
        plt.ylabel("y (normalized)")
        if title:
            plt.title(title)
        cbar = plt.colorbar(scatter)
        cbar.set_label("value (float)")
        plt.tight_layout()
        if show:
            plt.show()
