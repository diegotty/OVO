from typing import Dict, Any
import yaml
import json
from pathlib import Path
from pprint import pprint
from typing import Any
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import networkx as nx
from scene_graph.segment import Segment, SegmentGeometry, SegmentView, SegmentStore
from scene_graph.spatial_relations import geometry

def load_config(path: str) -> Dict[str, Any]:
    """
     stolen from OVO's utils @io_utils.py
    """
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)
    cfg = dict()
    update_recursive(cfg, cfg_special)
    return cfg

def update_recursive(dict1: Dict[str,Any], dict2: Dict[str,Any]) -> None:
    """ Recursively updates the first dictionary with the contents of the second dictionary.

    This function iterates through `dict2` and updates `dict1` with its contents. If a key from `dict2`
    exists in `dict1` and its value is also a dictionary, the function updates the value recursively.
    Otherwise, it overwrites the value in `dict1` with the value from `dict2`.

    Args:
        dict1: The dictionary to be updated.
        dict2: The dictionary whose entries are used to update `dict1`.

    Returns:
        None: The function modifies `dict1` in place.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def load_segments(scene_dir, min_points=1) -> SegmentStore:
    scene_dir = Path(scene_dir).expanduser().resolve()
    scene_file = scene_dir / "scene.json"
    with scene_file.open("r", encoding="utf-8") as file:
        scene_metadata = json.load(file)

    segment_ids_file = scene_dir / scene_metadata["segment_ids_file"]
    descriptors_file = scene_dir / scene_metadata["descriptors_file"]
    segment_ids = np.load(segment_ids_file).reshape(-1).astype(np.int64)
    descriptors = np.load(descriptors_file).astype(np.float32, copy=False)
    
    if len(segment_ids) != len(descriptors):
        raise RuntimeError("segment_ids.npy and descriptors.npy might not be matched")

    metadata_by_id = dict()
    for segment in scene_metadata["segments"]:
        segment_copy = segment.copy()
        segment_copy['views'] = []
        for view in segment.get('top_views', []):
            view_descriptor_file = scene_dir / view['descriptor']
            view_descriptor = np.load(view_descriptor_file).astype(np.float32, copy=False).reshape(-1)
            view_obj = SegmentView(int(view['keyframe_id']), view_descriptor, float(view['mask_area']))
            segment_copy['views'].append(view_obj)
        metadata_by_id[segment["id"]] = segment_copy

    segments = []
    for segment_id in segment_ids:
        segment_id = int(segment_id)
        if segment_id not in metadata_by_id:
            raise RuntimeError(f"segment {segment_id} pesent in segments_id.npy but not in scene.json")
        seg_metadata = metadata_by_id[segment_id]
        descriptor_row = int(seg_metadata["descriptor_row"])
        if not 0 <= descriptor_row < len(descriptors):
            raise RuntimeError(f"invalid descriptor row{descriptor_row} for segment {segment_id}")
        points_file = scene_dir / seg_metadata["points_file"]
        top_views = seg_metadata['views']
        keyframe_ids=set(
            int(kf_id)
            for kf_id in seg_metadata['keyframe_ids']
        )

        # valid points
        points = np.load(points_file)
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        # if the segment doesn't have enough points it gets skipped
        if len(points) < min_points:
            continue
        descriptor = descriptors[descriptor_row].reshape(-1).copy()
        segment_geometry_obj = geometry.compute_aabb(segment_id, points)
        segment_obj = Segment(
            id = segment_id,
            points = points,
            descriptor = descriptor,
            top_views = top_views,
            geometry = segment_geometry_obj,
            keyframe_ids = keyframe_ids

        )
        segments.append(segment_obj)
    return SegmentStore(segments)
