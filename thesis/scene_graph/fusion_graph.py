import networkx as nx
import numpy as np
import graph_utils
from typing import Any
from itertools import combinations

# --------------------- fusion affinity graph implementation
def point_coverage(pcd_a, pcd_b, th_dist):
    distances_ab = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    distances_ba = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))
    if len(distances_ab) == 0 or len(distances_ba) == 0:
        return 0.0
    coverage_ab = float(np.mean(distances_ab < th_dist))
    coverage_ba = float(np.mean(distances_ba < th_dist))
    return coverage_ab, coverage_ba

# th_dist: distance threshold (conservative)
# th_clip: semantic similarity threshold (conservative)
def compute_pair_fusion_affinity(node_a, node_b, th_dist, th_coverage, th_reciprocal):
    # if not 0.0 <= semantic_weight <= 1.0:
    #     raise ValueError('semantic weight must be between 0 an 1')
    distance, proximity = geometric_affinity(node_a['center'], node_b['center'], distance_scale)
    if proximity < th_dist:
        return None

    cosine, semantic_similarity = semantic_similarity(node_a['descriptor'], node_b['descriptor'])
    if semantic_similarity < th_clip:
        return None

    coverage_ab, coverage_ba = point_coverage(node_a['points'], node_b['points'], th_dist)
    max_coverage = max(coverage_ab, coverage_ba)
    min_coverage = min(coverage_ab, coverage_ba)
    can_fuse = (max_coverage > th_coverage and min_coverage > th_reciprocal)
    return {
        "centroid_distance" : proximity,
        "semantic_affinity" : semantic_similarity,
        "pcd_coverage" : max_coverage,
        "can_fuse": bool(can_fuse)
    }

def add_fusion_edges(graph):
    node_ids = list(graph.nodes)
    for node_a_id, node_b_id in combinations(node_ids, 2):
        node_a = graph.nodes[node_a_id]
        node_b = graph.nodes[node_b_id]

        can_fuse = compute_pair_fusion_affinity(node_a, node_b, )
        if can_fuse:
            graph.add_edge(node_a_id, node_b_id, centroid_distance=)

def build_fusion_graph(region, segments, semantic_weight, distance_scale, affinity_threshold):
    '''
    builds an graph with segments and nodes, and edges that define the combined similarity
    between a pair of nodes. edges are only created if the affinity exceeds affinity_threshold
    '''
    graph = nx.Graph()
    graph.graph['region_id'] = region['region_id']
    graph.graph['region_name'] = region['region_name']
    graph.graph['fusion_threshold'] = affinity_threshold
    add_segment_nodes(graph, region, segments)
    add_affinity_edges(graph, semantic_weight, distance_scale, affinity_threshold)
    return graph



