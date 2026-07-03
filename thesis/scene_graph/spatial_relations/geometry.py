from collections.abc import Iterable, Mapping
from typing import Any
import numpy as np


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

def compute_aabb( segment_id: int, points: np.ndarray) -> dict[str, Any]:
    """
    compute the geometry needed by the VLA-3D
    - `center` is the AABB center
    - `centroid` is the mean of the reconstructed segment points
    - `volume` is the bounding-box volume
    """
    points = _validate_points(points, segment_id=segment_id)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2.0
    # centroid = points.mean(axis=0)
    bbox = aabb_corners(bbox_min, bbox_max)

    return {
        "object_id": int(segment_id),
        "bbox_type": "aabb",
        "bbox": bbox,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "center": center.astype(np.float32, copy=False),
        # "centroid": centroid.astype(np.float32, copy=False),
        "size": size.astype(np.float32, copy=False),
        "volume": float(np.prod(size)),
        # "footprint_area": float(size[0] * size[1]),
        # "diagonal_length": float(np.linalg.norm(size)),
        # "nyu_id" : -1,
        # "nyu_label" : "unknown",
        # "raw_label" : "unknown"
    }


def compute_objects( segments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """
    convert parsed OVO segments into lightweight relation-extractor objects 
    """
    objects: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for segment in segments:
        segment_id = int(segment["id"])
        if segment_id in seen_ids:
            raise ValueError(f"duplicate segment ID: {segment_id}")
            continue
        seen_ids.add(segment_id)
        objects.append(compute_aabb(segment_id, segment["points"]))
    return objects


def build_region(objects: Iterable[Mapping[str, Any]], region_id: int = 0, region_name: str = "full_scene", ) -> dict[str, Any]:
    """
    build one region containing all relation objects 
   (suitable while OVO does not provide room-level region segmentation)
    """
    objects = list(objects)
    all_bbox_corners = []
    for obj in objects:
        all_bbox_corners.append(obj["bbox"])
    all_bbox_corners = np.concatenate(all_bbox_corners,axis=0)
    region_geometry = compute_aabb(region_id, all_bbox_corners)
    return {
        "region_id": int(region_id),
        "region_name": region_name,
        "region_bbox": region_geometry["bbox"],
        # "bbox_min": region_geometry["bbox_min"],
        # "bbox_max": region_geometry["bbox_max"],
        # "center": region_geometry["center"],
        # "size": region_geometry["size"],
        # "volume": region_geometry["volume"],
        "objects": objects,
        "relationships": {},
    }
