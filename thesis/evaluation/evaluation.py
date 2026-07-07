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
    


import csv
from pathlib import Path


def save_stage_summary_csv( first_list: list[dict], second_list: list[dict], third_list: list[dict], output_file: Path) -> None:
    lengths = { len(first_list), len(second_list), len(third_list) }
    if len(lengths) != 1:
        raise ValueError(
            "The three lists must have the same length: "
            f"{len(first_list)}, "
            f"{len(second_list)}, "
            f"{len(third_list)}"
        )

    rows = []
    for index, dictionaries in enumerate( zip( first_list, second_list, third_list)):
        merged_row = {}

        for dictionary in dictionaries:
            for key, value in dictionary.items():
                if ( key in merged_row and merged_row[key] != value):
                    raise ValueError(
                        f"Conflicting value for key '{key}' "
                        f"in row {index}: "
                        f"{merged_row[key]!r} != {value!r}"
                    )
                merged_row[key] = value
        rows.append(merged_row)

    if not rows:
        return

    # Keep columns in their first encountered order.
    field_names = []
    for row in rows:
        for key in row:
            if key not in field_names:
                field_names.append(key)

    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir( parents=True, exist_ok=True)

    with output_file.open( "w", newline="", encoding="utf-8") as file: 
        writer = csv.DictWriter( file, fieldnames=field_names,)
        writer.writeheader()
        writer.writerows(rows)
