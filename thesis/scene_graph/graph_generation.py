import networkx as nx
import numpy as np
from typing import Any
from itertools import combinations

def cosine_similarity( descriptor_a, descriptor_b):
    norm_a = np.linalg.norm(descriptor_a)
    norm_b = np.linalg.norm(descriptor_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # normalizing my norm to remove magnitude
    similarity = np.dot(descriptor_a, descriptor_b) / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))

def semantic_similarity(descriptor_a, descriptor_b):
    cosine = cosine_similarity(descriptor_a, descriptor_b)

    # from [-1, 1] to [0, 1] (but w/issue)
    #affinity = (cosine + 1.0) / 2.0

    #conservative shift
    affinity = max(0, cosine)
    return cosine, affinity

# could change to bbox-aware distance instead of centroid distance
# turning distance into proximity (distance behaves in wrong direction)
def geometric_affinity(center_a, center_b, distance_scale):
    if distance_scale <= 0.0:
        raise ValueError('distance_scale must be > 0  ......')
    distance = float(np.linalg.norm(center_a - center_b))
    proximity = float(np.exp(-distance / distance_scale))
    return distance, proximity

#semantic_weight is the lerp parameter. semantic_weight = 1: only semantic similarity
def compute_pair_affinity(node_a, node_b, semantic_weight, distance_scale):
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError('semantic weight must be between 0 an 1')
    cosine, semantic_similarity = semantic_similarity(node_a['descriptor'], node_b['descriptor'])
    distance, proximity = geometric_affinity(node_a['center'], node_b['center'], distance_scale)

    # weighted OR
    combined_affinity = semantic_weight * semantic_similarity + (1 - semantic_weight) * proximity

    # weighted AND
    combined_affinity = pow(semantic_similarity, semantic_weight) * pow(proximity, 1 - semantic_weight)
    # combined_affinity = semantic_similarity * proximity # basic AND
    return float(combined_affinity)

def add_affinity_edges(graph, semantic_weight, distance_scale, affinity_threshold):
    if not 0.0 <= affinity_threshold <= 1.0:
        raise ValueError('affinity treshold should be in [0, 1]')
    node_ids = list(graph.nodes)
    # combination returns all unordered combinations
    for node_a_id, node_b_id in combinations(node_ids, 2):
        node_a = graph.nodes[node_a_id]
        node_b = graph.nodes[node_b_id]

        pair_affinity = compute_pair_affinity(node_a, node_b, semantic_weight, distance_scale)
        if pair_affinity >= affinity_threshold:
            graph.add_edge(node_a_id, node_b_id, affinity=pair_affinity)

def add_spatial_edges(graph, region, spatial_relations):
    for relation in spatial_relations:
        anchors = region['relationships'].get(relation, {})

        for anchor_id, target_ids in anchors.items():
            for target_id in target_ids:
                graph.add_edge(target_id, anchor_id, relation=relation)
    
def add_segment_nodes(graph, region, segments):
    segments_by_id = []
    for segment in segments:
        segments_by_id[segment['id']] = segment

    for obj in region['objects']:
        segment_id = obj['object_id']

        segment = segments_by_id[segment_id]
        graph.add_node(
            segment_id,
            descriptor = np.asarray(segment['descriptor']).copy(),
            center = np.asarray(obj['center']).copy(),
            # point_cloud = int(len(segment['points']))
            # points_file = str(segment['points_file])
            # could add bbox stuff, but should we ?
        )

def build_affinity_graph(region, segments, semantic_weight, distance_scale, affinity_threshold):
    '''
    builds an graph with segments and nodes, and edges that define the combined similarity
    between a pair of nodes. edges are only created if the affinity exceeds affinity_threshold
    '''
    graph = nx.Graph()
    graph.graph['region_id'] = region['region_id']
    graph.graph['region_name'] = region['region_name']
    graph.graph['semantic_weight'] = semantic_weight
    graph.graph['distance_scale'] = distance_scale
    graph.graph['affinity_threshold'] = affinity_threshold
    add_segment_nodes(graph, region, segments)
    add_affinity_edges(graph, semantic_weight, distance_scale, affinity_threshold)
    return graph


def build_spatial_graph(graph, nodes, region, spatial_relations):
    '''
    build a spatial relations graph using affinity_graph's edges 
    '''
    graph.graph['region_id'] = region['region_id']
    graph.graph['region_name'] = region['region_name']
    graph.graph['relation_set'] = spatial_relations
    graph.graph['fragment_refined'] = False
    add_spatial_edges(graph, region, spatial_relations)
    return graph

