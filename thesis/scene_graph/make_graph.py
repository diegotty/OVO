from pathlib import Path
from spatial_graph import SpatialGraph
from fusion_graph import FusionGraph
from utils import ai_utils, load_utils, graph_utils

# OVO/thesis/scene_graph/
SCRIPT_DIR = Path(__file__).resolve().parent

# OVO/
REPO_ROOT = SCRIPT_DIR.parent.parent

def make_scene():
    config_path = SCRIPT_DIR / "config.yaml"
    config = load_utils.load_config(config_path)

    input_path = REPO_ROOT / config["input_path"]
    output_folder = REPO_ROOT / config["output_folder"]

    # load segments from neutral export (segmen
    segment_store = load_utils.load_segments(input_path, config.get("min_segment_points", 1))

    fusion_thresholds = config['fusion'].copy()
    fusion_thresholds['top_k_views'] = config['top_k_views']
    fuse_graph = FusionGraph(segment_store, fusion_thresholds)

    spatial_thresholds = config['spatial_graph'].copy()
    spatial_relations = []
    for relation, value in config['spatial_graph']['relations'].items():
        if value:
            spatial_relations.append(relation)

    spatial_graph = SpatialGraph(segment_store, spatial_relations, spatial_thresholds)
    update_graphs(segment_store, fuse_graph, spatial_graph)

def update_persistency(segment_store, persistence_threshold):
    for segment in segment_store.segments():
        if len(segment.keyframe_ids) < persistence_threshold:
            segment.persistent = False
 

def update_graphs(segment_store, fuse_graph, spatial_graph, persistence_threshold):
    updates = fuse_graph.update_graph()
    update_persistency(segment_store, persistence_threshold)
    spatial_graph.update_graph(updates)
    

if __name__ == "__main__":
    make_scene()
