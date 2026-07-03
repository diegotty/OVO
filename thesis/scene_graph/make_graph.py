from pathlib import Path
import networkx as nx
from spatial_relations import relation_extraction
import spatial_graph
from spatial_relations import geometry
import fusion_graph
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

    # load segments from neutral export
    segments = load_utils.load_segments(config["input_path"], config.get("min_segment_points", 1))
    
    # make new lightweight objects
    objects = geometry.compute_objects(segments)

    # build regions (compute geometry for objects)
    region = geometry.build_region(objects)
    ai_utils.print_region_summary(region)

    fusion_thresholds = config['fusion'].copy()
    fusion_thresholds['k_top_views'] = config['k_top_views']

    fuse_graph = fusion_graph.FusionGraph(region, segments, fusion_thresholds)
    spatial_graph = None
    

    # region is modified in place, returns list of relation classes
    spatial_relations = relation_extraction.compute_spatial_relations(config, region)
    ai_utils.print_relations(region)

    # spatial relations graph
    # graph = nx.MultiDiGraph()
    # nodes = graph_utils.add_segment_nodes(graph, region,segments)

    # spatial_relations_graph = spatial_graph.build_spatial_graph(graph, nodes, region, spatial_relations)
    
    # prints&validation utils
    ai_utils.print_relation_graph_summary(spatial_relations_graph)
    ai_utils.print_relation_graph_edges(spatial_relations_graph)
    ai_utils.validate_relation_graph(spatial_relations_graph)

def update_graphs(fuse_graph, spatial_relations):
    update_results = fuse_graph.update_graph()
    

if __name__ == "__main__":
    make_scene()
    if spatial_graph is None:
        spatial_graph = 

