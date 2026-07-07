from dataclasses import dataclass
from networkx import strongly_connected_components
from thesis.scene_graph.segment import SegmentStore
from thesis.scene_graph.utils import load_utils
from thesis.evaluation import replica_gt
from thesis.evaluation import replica_matching
from thesis.evaluation import replica_matching
from thesis.evaluation import utils
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

@dataclass
class Evaluator:
    def __init__(self, scene):
        self.scene = scene
        config_path = SCRIPT_DIR / "replica.yaml"
        self.config = load_utils.load_config(config_path)
        scene_name = scene + "_instances.json"
        instances_dir = REPO_ROOT / "data/evaluation/replica" / scene_name
        self.gt_instances = replica_gt.load_prepared_instances(instances_dir)
        self.gt_instances_summary = []
        self.gt_classes_summary = []


# loads segments from OUTPUT EXPORTS, and assigns to each a GT class & GT object instance AND
# SAVES THEM  to data/evaluation/replica/<scene_name>_segment_matches.csv
    def compute_instance_matches(self, matches, stage : str):
        # utils.print_match_summary(matches, config['ignored_classes'])
        gt_instances_row = utils.print_gt_fragmentation_summary(matches, self.config['ignored_classes'])
        self.gt_instances_summary.append(gt_instances_row)

    def compute_class_matches(self, segments):
        matches = replica_matching.match_segments_to_instances(segments=segments, instances=self.gt_instances)
        gt_classes_row = replica_matching.build_classes_summary_row(matches)
        utils.print_match_summary(matches, self.config['ignored_classes'])
        self.gt_classes_summary.append(gt_classes_row)
        return matches

    def save_matches(self, scene, matches, name=None):
        if name is not None:
            csv_name = scene + "_" + name +  "_segment_matches.csv"
        else:
            csv_name = scene + "_segment_matches.csv"
        dest_dir = REPO_ROOT / "data/evaluation/replica" / csv_name
        replica_matching.save_matches_csv(matches=matches, output_file=dest_dir)
        print(f'saved matches to {dest_dir} !')
    

