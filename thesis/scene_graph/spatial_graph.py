import networkx as nx
import numpy as np
from typing import Any
from thesis.scene_graph.spatial_relations import relation_extractor
from itertools import combinations

class SpatialGraph:
    def __init__(self, segment_store, spatial_relations, thresholds):
        self.thresholds = thresholds
        self.graph = nx.MultiDiGraph()
        self.segment_store = segment_store
        self.spatial_relations = spatial_relations
        for segment in segment_store.segments(not_absorbed_only=True):
            self.graph.add_node(segment.id)
        self._init_edges()

        # relation_extractor.compute_spatial_relations(thresholds, region, spatial_relations)
    def _add_edge(self, anchor, target, relation):
        self.graph.add_edge(target, anchor, key=relation)

    def _init_edges(self):
        for anchor in self.segment_store.segments(not_absorbed_only=True):
            # function accesses segment_store as read-only !!
            relations = relation_extractor.compute_spatial_relations(anchor, self.segment_store, self.spatial_relations, self.thresholds)
            for relation, targets in relations.items():
                for target in targets:
                    self._add_edge(anchor.id, target, relation)

    def _add_spatial_edges(self, region, spatial_relations):

        for relation in spatial_relations:
            anchors = region['relationships'].get(relation, {})
    
            for anchor_id, target_ids in anchors.items():
                for target_id in target_ids:
                    self.graph.add_edge(target_id, anchor_id, relation=relation)

    def rebuild(self):
        self.graph.clear()
    
        confirmed_segments = [segment for segment in self.segment_store.segments(not_absorbed_only=True)]

        for segment in self.segment_store.segments(not_absorbed_only=True):
            self.graph.add_node(segment.id)
        for anchor in confirmed_segments:
            relations = relation_extractor.compute_spatial_relations(anchor, self.segment_store, self.spatial_relations, self.thresholds)
            for relation, targets in relations.items():
                for target in targets:
                    self._add_edge(anchor.id, target, relation)


#    def update_graph(self, updates):
#        for absorbed in updates['absorbed']:
#            self.graph.remove_node(absorbed)
#        for anchor in updates['survivors']:
#            self.graph.remove_edges_from(list(self.graph.edges(anchor)))
#
#            # re-calculate relations
#            relations = relation_extractor.compute_spatial_relations(anchor, self.segment_store, self.spatial_relations, self.thresholds)
#            for relation, targets in relations.items():
#                for target in targets:
#                    self._add_edge(anchor.id, target.id, relation)
    

    # can need a rebuild because of fusion OR persistence updates
    def update_graph(self, updates):
        self.rebuild()
