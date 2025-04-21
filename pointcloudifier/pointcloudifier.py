"""
Utility to convert a 2D image into an (x, y, value) point cloud.

- x and y are mapped to [-1, 1] with the image centre as origin while the
  aspect ratio of the input is preserved.
- value is a float in the range 0..255. The default is the mean of the
  R, G and B channels. Pass custom weights (wr, wg, wb) to use a different
  grayscale conversion.
- The attributes ``dx``, ``dy`` and ``dv`` store the regular spacing of the
  point cloud along x, y and value. They are derived in *cloudify_image()*
  and updated whenever the value axis is rescaled or quantised. They remain
  unchanged if points are merely dropped.
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
    """Return the channel weights as a normalised NumPy array."""
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
        # The list stores tuples in the form (x, y, value).
        self.points: List[Tuple[float, float, float]] = []
        # Regular spacings in normalised units.
        self.dx: Optional[float] = None
        self.dy: Optional[float] = None
        self.dv: Optional[float] = None

    def cloudify_image(
        self,
        filename: str | Path,
        *,
        sample_rate: int = 1,
        weights: Optional[Weights] = None,
    ) -> List[Tuple[float, float, float]]:
        """Convert *filename* into a point cloud and determine regular spacings."""
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

        # Convert RGB to grayscale with optional weights.
        if weights is None:
            values = pixels.mean(axis=1, dtype=np.float32)
        else:
            w_arr = _validate_weights(weights)
            values = (pixels * w_arr).sum(axis=1, dtype=np.float32)
        values = np.clip(values, 0.0, 255.0)

        # Map pixel coordinates to [-1, 1] while keeping the aspect ratio.
        max_dim = max(w, h)
        norm_x = 2.0 * coords_x / (max_dim - 1) - w / max_dim
        norm_y = -(2.0 * coords_y / (max_dim - 1) - h / max_dim)

        self.points = list(zip(norm_x.tolist(), norm_y.tolist(), values.tolist()))

        # Determine regular spacings.
        step_norm = 2.0 * sample_rate / (max_dim - 1)
        self.dx = step_norm
        self.dy = step_norm
        self.dv = 1.0  # Original pixel intensities differ by at least one.
        return self.points

    def remap_values(self, new_min: float, new_max: float) -> List[Tuple[float, float, float]]:
        """Linearly map intensities into the range [new_min, new_max] and update dv."""
        if not self.points:
            raise RuntimeError("No point cloud available. Generate or load one first.")
        if new_min > new_max:
            raise ValueError("new_min must be less than or equal to new_max")

        xs, ys, vals = zip(*self.points)
        old_min = min(vals)
        old_max = max(vals)

        if old_min == old_max:
            midpoint = (new_min + new_max) / 2.0
            mapped_vals = [midpoint] * len(vals)
            scale = 0.0
        else:
            scale = (new_max - new_min) / (old_max - old_min)
            mapped_vals = [new_min + (v - old_min) * scale for v in vals]

        self.points = list(zip(xs, ys, mapped_vals))
        # Update dv if available.
        if self.dv is not None:
            self.dv *= scale
        return self.points

    def quantise_values(self, palette: List[float]) -> List[Tuple[float, float, float]]:
        """Snap every intensity to the nearest entry in *palette* and set dv.

        The method first remaps the current intensities to the range defined by
        the first and last palette values to avoid unintended clipping. After
        the remap each value is replaced by the palette entry that is closest
        to it. The coordinates x and y are not modified.
        """
        if not self.points:
            raise RuntimeError("No point cloud available. Generate or load one first.")
        if len(palette) < 2:
            raise ValueError("palette must contain at least two entries")

        self.remap_values(palette[0], palette[-1])

        palette_arr = np.asarray(palette, dtype=np.float32)
        xs, ys, vals = zip(*self.points)
        vals_arr = np.asarray(vals, dtype=np.float32).reshape(-1, 1)

        # Find the nearest palette entry for each remapped value.
        diff = np.abs(vals_arr - palette_arr.reshape(1, -1))
        nearest_indices = diff.argmin(axis=1)
        snapped_vals = palette_arr[nearest_indices].tolist()

        self.points = list(zip(xs, ys, snapped_vals))
        # Update dv to the smallest distance between consecutive palette values.
        sorted_palette = np.sort(palette_arr)
        diffs = np.diff(sorted_palette)
        self.dv = float(diffs.min()) if diffs.size > 0 else None
        return self.points

    def drop_below(self, threshold: float) -> List[Tuple[float, float, float]]:
        """Remove all points with a value below *threshold*."""
        if not self.points:
            raise RuntimeError("No point cloud available. Generate or load one first.")

        self.points = [pt for pt in self.points if pt[2] >= threshold]
        return self.points

    def drop_above(self, threshold: float) -> List[Tuple[float, float, float]]:
        """Remove all points with a value above *threshold*."""
        if not self.points:
            raise RuntimeError("No point cloud available. Generate or load one first.")

        self.points = [pt for pt in self.points if pt[2] <= threshold]
        return self.points

    def drop_random(
        self, fraction: float = 0.1, *, seed: Optional[int] = None
    ) -> List[Tuple[float, float, float]]:
        """Remove a fraction of points uniformly at random. Spacings stay unchanged."""
        if not self.points:
            raise RuntimeError("No point cloud available. Generate or load one first.")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be between 0.0 and 1.0")
        if fraction == 0.0:
            return self.points
        if fraction == 1.0:
            self.points = []
            return self.points

        rng = np.random.default_rng(seed)
        n_total = len(self.points)
        n_remove = int(np.floor(n_total * fraction))
        indices_to_remove = rng.choice(n_total, size=n_remove, replace=False)
        mask = np.ones(n_total, dtype=bool)
        mask[indices_to_remove] = False
        self.points = [pt for keep, pt in zip(mask, self.points) if keep]
        return self.points

    def shake(
        self,
        shake_x: float = 0.0,
        shake_y: float = 0.0,
        shake_v: float = 0.0,
        *,
        seed: Optional[int] = None,
    ) -> List[Tuple[float, float, float]]:
        """Perturb every point by a random offset proportional to the spacings.

        The parameters *shake_x*, *shake_y* and *shake_v* must lie in the range
        [-1, 1]. A value of 1 means that the corresponding coordinate can shift
        by up to plus or minus one full grid spacing. The regular spacing
        attributes dx, dy and dv are intentionally kept unchanged.
        """
        if not self.points:
            raise RuntimeError("No point cloud available. Generate or load one first.")
        for name, val in {"shake_x": shake_x, "shake_y": shake_y, "shake_v": shake_v}.items():
            if not -1.0 <= val <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")

        if self.dx is None or self.dy is None or self.dv is None:
            raise RuntimeError("Spacings dx, dy, dv have not been initialised.")

        rng = np.random.default_rng(seed)
        xs, ys, vals = zip(*self.points)
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        vals = np.asarray(vals, dtype=np.float32)

        xs += rng.uniform(-self.dx * shake_x, self.dx * shake_x, size=xs.shape)
        ys += rng.uniform(-self.dy * shake_y, self.dy * shake_y, size=ys.shape)
        vals += rng.uniform(-self.dv * shake_v, self.dv * shake_v, size=vals.shape)

        self.points = list(zip(xs.tolist(), ys.tolist(), vals.tolist()))
        return self.points

    def save_json(self, filepath: str | Path, *, indent: int = 2) -> None:
        """Save the current point cloud to *filepath* as JSON."""
        if not self.points:
            raise RuntimeError("No point cloud to save. Run cloudify_image() first.")
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(
                {"points": self.points, "dx": self.dx, "dy": self.dy, "dv": self.dv},
                fh,
                indent=indent,
            )

    def load_json(self, filepath: str | Path) -> List[Tuple[float, float, float]]:
        """Load a point cloud that was previously created with save_json()."""
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.points = data["points"]
        self.dx = data.get("dx")
        self.dy = data.get("dy")
        self.dv = data.get("dv")
        return self.points

    def plot(
        self,
        *,
        cmap: str = "viridis",
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
