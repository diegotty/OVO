# --------------------- standalone affinity graph implementation
#semantic_weight is the lerp parameter. semantic_weight = 1: only semantic similarity
def compute_pair_affinity(node_a, node_b, semantic_weight, distance_scale):
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError('semantic weight must be between 0 an 1')
    cosine, semantic_similarity = graph_utils.semantic_similarity(node_a['descriptor'], node_b['descriptor'])
    distance, proximity = graph_utils.geometric_affinity(node_a['center'], node_b['center'], distance_scale)

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
    graph_utils.add_segment_nodes(graph, region, segments)
    add_affinity_edges(graph, semantic_weight, distance_scale, affinity_threshold)
    return graph
# ---------------------

