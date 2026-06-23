from pathlib import Path
import relation_extraction
import geometry
import utils

def make_scene(scene_path="", debug_info = False):
    config_path = Path(scene_path) / "config.yaml"
    config = utils.load_config(config_path.as_posix())

    #load segments from neutral export
    segments = utils.load_segments(config["scene_graph"]["input_path"], config["scene_graph"].get("min_segment_points", 1))
    
    objects = geometry.compute_objects(segments)
    region = geometry.build_region(objects)
    relation_extraction.compute_spatial_relationships(config, region)

    for relation, anchors in region["relationships"].items():
        count = sum(len(targets) for targets in anchors.values())
        print(f"{relation}: {count}")
