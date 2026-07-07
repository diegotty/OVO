from thesis.scene_graph.utils import load_utils
from thesis.evaluation import replica_gt
from thesis.evaluation import replica_matching
from thesis.evaluation import replica_matching
from thesis.evaluation import utils
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# loads segments from OUTPUT EXPORTS, and assigns to each a GT class & GT object instance AND
# SAVES THEM  to data/evaluation/replica/<scene_name>_segment_matches.csv
def calculate_matches(scene, final_segments = None):
    config_path = SCRIPT_DIR / "replica.yaml"
    config = load_utils.load_config(config_path)

    # root of output_export.py
    export_dir = REPO_ROOT / "exported" / scene
    #func doenst save anything ... just lods
    segment_store = load_utils.load_segments(export_dir, config.get('min_segment_points',0))
    scene_name = scene + "_instances.json"
    instances_dir = REPO_ROOT / "data/evaluation/replica" / scene_name
    gt_instances = replica_gt.load_prepared_instances(instances_dir)
    segments = list(segment_store.segments())

    # to do the same thing but with the AFTER-FUSION segments
    if final_segments is not None:
        segments = final_segments
    print(f'number of segments: {len(segments)}')

    matches = replica_matching.match_segments_to_instances(segments=segments, instances=gt_instances)
    # utils.print_match_summary(matches, config['ignored_classes'])
    utils.print_gt_fragmentation_summary(matches, config['ignored_classes'])

    return matches

def save_matches(scene, matches, name=None):
    if name is not None:
        csv_name = scene + "_" + name +  "_segment_matches.csv"
    else:
        csv_name = scene + "_segment_matches.csv"
    dest_dir = REPO_ROOT / "data/evaluation/replica" / csv_name
    replica_matching.save_matches_csv(matches=matches, output_file=dest_dir)
    print(f'saved matches to {dest_dir} !')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dir should be ovo/data/input/Replica/<scene_name>
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    matches = calculate_matches(args.scene) 
    save_matches(args.scene, matches)
