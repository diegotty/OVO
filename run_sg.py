from thesis.export import output_exporter
from thesis.scene_graph.make_graph import Controller
from thesis.scene_graph.utils import validation
from thesis.evaluation import evaluation, fusion_metrics
from thesis.evaluation.utils import print_fusion_metrics
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()

    #OVO output dir
    parser.add_argument("--input_dir", type=Path, required=True)

    # completed vaina dir
    parser.add_argument("--export_dir", type=Path, required=True)
    args = parser.parse_args()
    args.export_dir.mkdir(parents=True, exist_ok=True)
    print('--- export stage ---')
    print(f'input: {args.input_dir}')
    print(f'export: {args.export_dir}')
    # export_status = output_exporter.extract(input_dir=args.input_dir, output_dir=args.export_dir)
    # print(f'export result: {export_status!r}')

    print('--- scene graph stage ---')
    controller = Controller(args.export_dir, args.export_dir)
    initial_active_count = len(list(controller.segment_store.segments(not_absorbed_only=True)))
    # validation.validate_segment_store(controller.segment_store, stage='initial')
    # validation.validate_fusion_graph(controller.fusion_graph, stage='before_fusion')
    # validation.validate_spatial_graph(controller.spatial_graph)
    matches = evaluation.calculate_matches()
    # final_segments = {
    #    segment.id : []
    #    for segment in controller.segment_store.segments(not_absorbed_only=True)
    # }
    # initial_results = fusion_metrics.evaluate_fusion(matches, final_segments)
    # print_fusion_metrics(initial_results)
    
    print('--- fusing !!! ---')
    fusion_map, fusion_parts = controller.update_graphs()
    # validation.validate_segment_store(controller.segment_store, stage='final')
    # validation.validate_fusion_updates(controller.segment_store, controller.fusion_graph, fusion_parts, initial_active_count)
    # validation.validate_fusion_graph(controller.fusion_graph, stage='after fusion')
    # validation.validate_spatial_graph(controller.spatial_graph)

    final_segments = {
        segment.id : []
        for segment in controller.segment_store.segments(not_absorbed_only=True)
    }
    for absorbed, survivor in fusion_map.items():
        if survivor in fusion_parts['survivors']:
            final_segments[survivor].append(absorbed)
    results = fusion_metrics.evaluate_fusion(matches, final_segments)
    print_fusion_metrics(results)
    evaluation.calculate_matches(list(controller.segment_store.segments(not_absorbed_only=True)))

    


if __name__ == "__main__":
    main()
