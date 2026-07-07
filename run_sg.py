from thesis.export import output_exporter
from thesis.scene_graph.graph_controller import Controller
from thesis.scene_graph.utils.validation import Validation
from thesis.evaluation import evaluation, fusion_metrics
from thesis.evaluation.utils import print_fusion_metrics
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# run from ovo/
# ONE SCENE AT A TIME
def main():
    parser = argparse.ArgumentParser()
    # dir should be ovo/data/input/Replica/<scene_name>
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--test_name", type=str, required=True)

    args = parser.parse_args()
    # should be ovo/evaluation/<run_name>
    input_dir = SCRIPT_DIR / "data/output/Replica/" / args.scene
    output_dir = SCRIPT_DIR / "evaluation" / args.test_name / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)
    print('--- export stage ---')
    print(f'scene dir: {input_dir}')
    print(f'output_dir: {output_dir}')

    output_exporter.extract(input_dir)
    # print(f'export result: {export_status!r}')
    export_dir = SCRIPT_DIR / "exported" / args.scene
    controller = Controller(export_dir)
    print('loaded OVO segments')
    initial_active_count = len(list(controller.segment_store.segments(not_absorbed_only=True)))
    validator = Validation(
        flags={
            'segment_store' : True,
            'fusion_graph' : False,
            'spatial_graph' : False
        },
        segment_store=controller.segment_store,
        fusion_graph=controller.fusion_graph,
        spatial_graph=controller.spatial_graph,
        initial_active_count=initial_active_count
    )

    print('--- STARTING STAGE ---')
    validator.validate('initial')
    matches = evaluation.calculate_matches(args.scene)

    # initial_segments = {segment.id : [] for segment in controller.segment_store.segments(not_absorbed_only=True)}
    # initial_results = fusion_metrics.evaluate_fusion(matches, initial_segments)
    # print_fusion_metrics(initial_results)
    
    print('--- FUSION STAGE ---')
    fusion_map, final_clusters = controller.fusion_graph.update_graph()
    validator.validate('after fusion')
    validator.validate_fusion_updates(final_clusters)
    results = fusion_metrics.evaluate_fusion(matches=matches, final_clusters=final_clusters)
    print_fusion_metrics(results)
    evaluation.calculate_matches(args.scene, list(controller.segment_store.segments(not_absorbed_only=True)))


    print(f'--- PERSISTENCE-BASED FILTERING ---')
    controller.persistence_filter()
    validator.validate('after filtering')
    evaluation.calculate_matches(args.scene, list(controller.segment_store.segments(confirmed_only=True)))
    
if __name__ == "__main__":
    main()
