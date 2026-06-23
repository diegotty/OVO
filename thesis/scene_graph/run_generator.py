from pathlib import Path
import relation_extraction
import geometry
import utils

def make_scene(scene_path="thesis/scene_graph/"):
    config_path = Path(scene_path) / "config.yaml"
    config = utils.load_config(config_path.as_posix())

    #load segments from neutral export
    segments = utils.load_segments(config["input_path"], config.get("min_segment_points", 1))
    
    objects = geometry.compute_objects(segments)
    region = geometry.build_region(objects)
    relations = relation_extraction.compute_spatial_relationships(config, region)
    print_region_summary(region)


    for relation, anchors in region["relationships"].items():
        count = sum(len(targets) for targets in anchors.values())
        print(f"{relation}: {count}")
from pprint import pprint

import numpy as np


def print_region_summary(region: dict) -> None:
    objects = region["objects"]

    print("\n=== REGION SUMMARY ===", flush=True)
    print("region_id:", region["region_id"], flush=True)
    print("region_name:", region["region_name"], flush=True)
    print("number of objects:", len(objects), flush=True)

    region_bbox = np.asarray(region["region_bbox"])
    print("region_bbox shape:", region_bbox.shape, flush=True)
    print("region_bbox:", flush=True)
    print(region_bbox, flush=True)

    print("\n=== OBJECTS ===", flush=True)

    for obj in objects:
        print(
            f"object_id={obj['object_id']} "
            f"center={np.asarray(obj['center'])} "
            f"size={np.asarray(obj['size'])} "
            f"volume={obj['volume']:.6f} "
            f"bbox_shape={np.asarray(obj['bbox']).shape} "
            f"nyu_id={obj.get('nyu_id')}",
            flush=True,
        )

    print("======================\n", flush=True)

if __name__ == "__main__":
    make_scene()
