from thesis.scene_graph.utils import load_utils
from thesis.evaluation import replica_gt
from thesis.evaluation import replica_matching
from thesis.evaluation import replica_matching
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
    segments.sort(key=lambda segment: len(segment.points), reverse=True)
    largest_segments = segments[:20]
    
    # pls work
    segment = largest_segments[0]
    print("segment ID:", segment.id)
    print("points shape:", segment.points.shape)
    print("first five points:")
    print(segment.points[:5])

    matches = replica_matching.match_segments_to_instances(segments=largest_segments, instances=gt_instances)
    for match in matches:
        print(
            f"segment={match.segment_id:<4} "
            f"GT={str(match.gt_instance_id):<4} "
            f"class={str(match.gt_class_name):<15} "
            f"coverage={match.best_coverage:.3f} "
            f"second={match.second_best_coverage:.3f} "
            f"status={match.status}"
        )
    dest_dir = REPO_ROOT / "data/evaluation/replica/office0_sanity_test.csv"
    replica_matching.save_matches_csv(matches=matches, output_file=dest_dir)

if __name__ == '__main__':
    main()
