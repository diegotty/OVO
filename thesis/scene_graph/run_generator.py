from pathlib import Path
import relation_extraction
import geometry
import utils

# OVO/thesis/scene_graph/
SCRIPT_DIR = Path(__file__).resolve().parent

# OVO/
REPO_DIR = SCRIPT_DIR.parent.parent

def make_scene():
    config_path = SCRIPT_DIR / "config.yaml"
    config = utils.load_config(config_path)

    input_path = REPO_ROOT / config["input_path"]
    output_folder = REPO_ROOT / config["output_folder"]

    #load segments from neutral export
    segments = utils.load_segments(config["input_path"], config.get("min_segment_points", 1))
    
    objects = geometry.compute_objects(segments)
    region = geometry.build_region(objects)
    utils.print_region_summary(region)
    relations = relation_extraction.compute_spatial_relationships(config, region)


    # for relation, anchors in region["relationships"].items():
    #     count = sum(len(targets) for targets in anchors.values())
    #     print(f"{relation}: {count}")
