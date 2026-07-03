import networkx as nx
import numpy as np
import graph_utils
from typing import Any
from spatial_relations import relation_extractor
from itertools import combinations

class SpatialGraph:
    def __init__(self, nodes, region, spatial_relations, thresholds):
        self.thresholds = thresholds
        self.graph = nx.MultiDiGraph()
        self.graph.graph['region_id'] = region['region_id']
        self.graph.graph['region_name'] = region['region_name']
        self.graph.graph['relation_set'] = spatial_relations
        self.spatial_relations = spatial_relations

        relation_extractor.compute_spatial_relations(thresholds, region, spatial_relations)
        self._add_spatial_edges(region, spatial_relations)

    def _add_spatial_edges(self, region, spatial_relations):
        for relation in spatial_relations:
            anchors = region['relationships'].get(relation, {})
    
            for anchor_id, target_ids in anchors.items():
                for target_id in target_ids:
                    self.graph.add_edge(target_id, anchor_id, relation=relation)

    def update_graph(self, update_results):
        compute_spatial_relations(thresholds)
    

