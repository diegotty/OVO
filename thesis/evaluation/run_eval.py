from thesis.scene_graph.utils import load_utils
from thesis.evaluation import replica_gt
from thesis.evaluation import replica_matching
from thesis.evaluation import replica_matching
from thesis.evaluation import utils
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
def main():
    config_path = SCRIPT_DIR / "replica.yaml"
    config = load_utils.load_config(config_path)
    export_dir = REPO_ROOT / "exported/office0"
    segment_store = load_utils.load_segments(export_dir, config.get('min_segment_points',0))
    instances_dir = REPO_ROOT / "data/evaluation/replica/office0_instances.json"
    gt_instances = replica_gt.load_prepared_instances(instances_dir)
    segments = list(segment_store.segments())
    print(f'number of segments: {len(segments)}')

    matches = replica_matching.match_segments_to_instances(segments=segments, instances=gt_instances)
    utils.print_match_summary(matches, config['ignored_classes'])

    dest_dir = REPO_ROOT / "data/evaluation/replica/office0_segment_matches.csv"
    replica_matching.save_matches_csv(matches=matches, output_file=dest_dir)
    print(f'saved matches to {dest_dir} !')

if __name__ == '__main__':
    main()
