from typing import Dict, Any
import yaml
import json
from pathlib import Path
from pprint import pprint
from typing import Any
import numpy as np
import networkx as nx

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

def load_segments(scene_dir, min_points=1):
    scene_dir = Path(scene_dir).expanduser().resolve()
    scene_file = scene_dir / "scene.json"
    
    with scene_file.open("r", encoding="utf-8") as file:
        scene_metadata = json.load(file)

    segment_ids_file = scene_dir / scene_metadata["segment_ids_file"]
    descriptors_file = scene_dir / scene_metadata["descriptors_file"]
    segment_ids = np.load(segment_ids_file).reshape(-1).astype(np.int64)
    descriptors = np.load(descriptors_file).astype(np.float32, copy=False)
    
    if len(segment_ids) != len(descriptors):
        raise RuntimeError("segment_ids.npy and descriptors.py might not be matched")

    metadata_by_id = dict()
    for segment in scene_metadata["segments"]:
        metadata_by_id[segment["id"]] = segment

    segments = []
    for segment_id in segment_ids:
        if segment_id not in metadata_by_id:
            raise RuntimeError(f"segment {segment_id} pesent in segments_id.npy but not in scene.json")
        seg_metadata = metadata_by_id[segment_id]
        descriptor_row = int(seg_metadata["descriptor_row"])
        points_file = scene_dir / seg_metadata["points_file"]
        points = np.load(points_file)
        # finite_mask = np.isfinite(points).all(axis=1)
        # points = points[finite_mask]
        if len(points) < min_points:
            continue
        descriptor = descriptors[descriptor_row].reshape(-1).copy()
        segments.append({
            "id" : segment_id,
            "points" : points,
            "points_file" : points_file,
            "descriptor" : descriptor,
            "descriptor_row" : descriptor_row,
        })
    return segments
