from pathlib import Path
import graph_generation as g_gen
import networkx as nx
import relation_extraction
import geometry
import utils

# OVO/thesis/scene_graph/
SCRIPT_DIR = Path(__file__).resolve().parent

# OVO/
REPO_ROOT = SCRIPT_DIR.parent.parent

def make_scene():
    config_path = SCRIPT_DIR / "config.yaml"
    config = utils.load_config(config_path)

    input_path = REPO_ROOT / config["input_path"]
    output_folder = REPO_ROOT / config["output_folder"]

    # load segments from neutral export
    segments = utils.load_segments(config["input_path"], config.get("min_segment_points", 1))
    
    # make new lightweight objects
    objects = geometry.compute_objects(segments)

    # build regions (compute geometry for objects)
    region = geometry.build_region(objects)
    utils.print_region_summary(region)

    # compute spatial relationships between objects
    affinity_graph = g_gen.build_affinity_graph(
        region = region, 
        segments = segments, 
        semantic_weight = config['semantic_weight'], 
        distance_scale = config['distance_scale'],
        affinity_threshold = config['affinity_threshold']
    )

    # region is modified in place, returns list of relation classes
    spatial_relations = relation_extraction.compute_spatial_relations(config, region)
    utils.print_relations(region)

    # spatial relations graph
    graph = nx.MultiDiGraph()
    nodes = g_gen.add_segment_nodes(graph, region,segments)

    # .... get fragmentation-robust nodes and pass them to build_spatial_graph

    spatial_relations_graph = g_gen.build_spatial_graph(graph, nodes, region, spatial_relations)
    
    # prints&validation utils
    utils.print_relation_graph_summary(spatial_relations_graph)
    utils.print_relation_graph_edges(spatial_relations_graph)
    utils.validate_relation_graph(spatial_relations_graph)

if __name__ == "__main__":
    make_scene()
