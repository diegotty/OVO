from thesis.export import output_exporter
from thesis.scene_graph.make_graph import Controller
from thesis.scene_graph import validation
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    # requires a path as an argument when executing the file
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, required=True)
    args = parser.parse_args()
    args.export_dir.mkdir(parents=True, exist_ok=True)
    print('--- export stage ---')
    print(f'input: {args.input_dir}')
    print(f'export: {args.export_dir}')
    export_status = output_exporter.extract(input_dir=args.input_dir, output_dir=args.export_dir)
    print(f'export result: {export_status!r}')
    print('--- scene graph stage ---')

    controller = Controller(args.export_dir, args.export_dir)
    initial_active_count = len(list(controller.segment_store.segments(not_absorbed_only=True)))
    validation.validate_segment_store(controller.segment_store, stage='initial')
    validation.validate_fusion_graph(controller.fusion_graph, stage='before_fusion')
    validation.validate_spatial_graph(controller.spatial_graph)
    
    print('--- fusing !!! ---')
    fusion_updates = controller.update_graphs()
    validation.validate_segment_store(controller.segment_store, stage='final')
    validation.validate_fusion_graph(controller.fusion_graph, stage='after fusion')
    validation.validate_fusion_updates(controller.segment_store, controller.fusion_graph, fusion_updates, initial_active_count)
    validation.validate_spatial_graph(controller.spatial_graph)


if __name__ == "__main__":
    main()
