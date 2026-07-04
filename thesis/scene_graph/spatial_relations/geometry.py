from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping
from typing import Any
import numpy as np

@dataclass
class SegmentGeometry:
    aabb_min: np.ndarray
    aabb_max: np.ndarray

    # mayb one day
    obb_center: np.ndarray | None = None
    obb_extent: np.ndarray | None = None
    obb_rotation: np.ndarray | None = None

    def center(self) -> np.ndarray:
        return (np.asarray(self.aabb_min, dtype=np.float32) + np.asarray(self.aabb_max, dtype=np.float32)) * 0.5

def _validate_points(points: np.ndarray, segment_id) -> np.ndarray:
    points = np.asarray(points)
    prefix = f"segment {segment_id}: " 
    if len(points) == 0:
        raise RuntimeError(f"{prefix}empty pointcloud !")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError( f"{prefix}expected points with shape [N, 3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{prefix}point cloud contains non-finite coordinates")
    return points.astype(np.float32, copy=False)


def aabb_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """
    build the eight corners of an axis-aligned bounding box
    """
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    return np.array(
        [
            [xmin, ymax, zmax],
            [xmax, ymax, zmax],
            [xmax, ymin, zmax],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmax, ymin, zmin],
            [xmin, ymin, zmin],
        ],
        dtype=np.float32,
    )

def compute_aabb( segment_id: int, points: np.ndarray) -> SegmentGeometry:
    """
    compute the geometry needed by the VLA-3D
    - `center` is the AABB center
    - `centroid` is the mean of the reconstructed segment points
    - `volume` is the bounding-box volume
    """
    points = _validate_points(points, segment_id=segment_id)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    # size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2.0
    # centroid = points.mean(axis=0)
    # bbox = aabb_corners(bbox_min, bbox_max)
    return SegmentGeometry(aabb_min=bbox_min, aabb_max=bbox_max)
