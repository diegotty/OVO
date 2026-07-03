import networkx as nx
import open3d as o3d
import numpy as np
from utils import graph_utils
from typing import Any
from utils import graph_utils
from itertools import combinations
from spatial_relations import geometry
import heapq

class FusionGraph:
    '''
    builds an graph with segments and nodes, and edges that define the combined similarity
    between a pair of nodes. edges are only created if the affinity exceeds affinity_threshold
    '''
    def __init__(self, region, segments, thresholds, debug=False):
        # TODO: will hae to make this modular 
        self.thresholds = thresholds
        self.fuse_queue = []
        self.debug = debug
        self.graph = nx.Graph()
        self.graph.graph['region_id'] = region['region_id']
        self.graph.graph['region_name'] = region['region_name']
        graph_utils.add_segment_nodes(self.graph, region, segments)
        self._init_edges()
        

    def point_coverage(self, points_a, points_b, point_dist_th):

        pcd_a = o3d.geometry.PointCloud()
        pcd_a.points = o3d.utility.Vector3dVector(points_a)
        pcd_b = o3d.geometry.PointCloud()
        pcd_b.points = o3d.utility.Vector3dVector(points_b)
        distances_ab = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
        distances_ba = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))
        if len(distances_ab) == 0 or len(distances_ba) == 0:
            return 0.0, 0
        coverage_ab = float(np.mean(distances_ab < point_dist_th))
        coverage_ba = float(np.mean(distances_ba < point_dist_th))
        return coverage_ab, coverage_ba
    
    # semantic_th: semantic similarity threshold (OVO param)
    # coverage_th: pcd coverage % threshold
    def _compute_pair_fusion_affinity(self, node_a, node_b):
        distance, proximity = graph_utils.geometric_affinity(node_a['center'], node_b['center'], self.thresholds['distance_scale'])
        if distance > self.thresholds['distance_threshold']:
            return None

        cosine, semantic_similarity = graph_utils.semantic_similarity(node_a['descriptor'], node_b['descriptor'])
        if semantic_similarity < self.thresholds['semantic_threshold']:
            return None
    
        coverage_ab, coverage_ba = self.point_coverage(node_a['points'], node_b['points'], self.thresholds['point_distance_threshold'])
        max_coverage = max(coverage_ab, coverage_ba)
        # min_coverage = min(coverage_ab, coverage_ba)

        proximity_score = np.clip((self.thresholds['distance_threshold'] - distance) / ( self.thresholds['distance_threshold']), 0.0, 1.0)
        semantic_score = np.clip((semantic_similarity - self.thresholds['semantic_threshold']) / (1.0 - self.thresholds['semantic_threshold']), 0.0, 1.0)
        coverage_score = np.clip((max_coverage - self.thresholds['coverage_threshold']) / (1.0 - self.thresholds['coverage_threshold']), 0.0, 1.0)

        # TODO, FIX: use good condition
        can_fuse = (max_coverage > self.thresholds['coverage_threshold'])
        # geometric mean
        fuse_score = (semantic_score * proximity_score * coverage_score) ** (1/3)
        return {
            # "centroid_distance" : proximity,
            # "semantic_affinity" : semantic_similarity,
            # "pcd_coverage" : max_coverage,
            'fuse_score' : fuse_score,
            'can_fuse': bool(can_fuse)
        }
    
    def _add_edge(self, node_a_id, node_b_id, affinity):
        print(f'added edge: ({node_a_id},{node_b_id}, with fuse score:  {affinity['fuse_score']}')
        self.graph.add_edge(node_a_id, node_b_id, 
            # centroid_distance = affinity['centroid_distance'],
            # semantic_affinity = affinity['semantic_affinity'],
            # pcd_coverage = affinity['pcd_coverage'],
            fuse_score = affinity['fuse_score'],
            can_fuse = affinity['can_fuse']
        )
        if affinity['can_fuse']:
            version_a = self.graph.nodes[node_a_id]['version']
            version_b = self.graph.nodes[node_b_id]['version']
            print(f'added to queue: ({node_a_id}, {node_b_id})')
            heapq.heappush(
                self.fuse_queue, 
                (-affinity['fuse_score'],
                 node_a_id, version_a,
                 node_b_id, version_b)
            )

    def _init_edges(self):
        node_ids = list(self.graph.nodes)
        for node_a_id, node_b_id in combinations(node_ids, 2):
            node_a = self.graph.nodes[node_a_id]
            node_b = self.graph.nodes[node_b_id]
    
            affinity = self._compute_pair_fusion_affinity(node_a, node_b)
            # distance_scale, dist_th, point_dist_th, semantic_th, coverage_th)
    
            if affinity is not None:
                self._add_edge(node_a_id, node_b_id, affinity)

    def pop_fusion(self):
        while self.fuse_queue:
            # node_a and node_b are ids, not objects
            (neg_fuse_score,
             node_a, q_version_a,
             node_b, q_version_b) = heapq.heappop(self.fuse_queue) 
            print(f'popped. will fuse {node_b} into {node_a}, with fuse_score={-neg_fuse_score}')

            # nodes not in graph
            if node_a not in self.graph or node_b not in self.graph:
                continue
            # edge got removed
            if not self.graph.has_edge(node_a, node_b):
                continue
            # wrong version
            if self.graph.nodes[node_a]['version'] != q_version_a or \
                self.graph.nodes[node_b]['version'] != q_version_b: 
                continue

            edge = self.graph.edges[node_a, node_b]
            if not edge["can_fuse"]:
                continue
            return node_a, node_b
        return None

    # TODO: should prob global
    def l1_medoid(self,descriptors):
        descriptors = np.asarray(descriptors, dtype=np.float32)
    
        pairwise_distances = np.abs(
            descriptors[:, None, :] - descriptors[None, :, :]
        ).sum(axis=2)
    
        best_idx = int(np.argmin(pairwise_distances.sum(axis=1)))
    
        return descriptors[best_idx].copy(), best_idx

    def _fuse_nodes(self, node_a_id, node_b_id):
        # TODO: for real-time implementation, add keyframes and update the heap of survivor
        node_a = self.graph.nodes[node_a_id]
        node_b = self.graph.nodes[node_b_id]
        node_a["points"] = np.concatenate([node_a["points"], node_b["points"]], axis=0)

        neighbors_a = set(self.graph.neighbors(node_a_id))
        neighbors_b = set(self.graph.neighbors(node_b_id))
        neighbors = neighbors_a.union(neighbors_b)
        neighbors_a.discard(node_a_id)

        #recompute geometry & descriptor of a
        geom = geometry.compute_aabb(node_a_id, node_a['points'])
        #recompute descriptor .....
        # merge and keep the first k_top_views based on mask_area

        merged_views = node_a["top_views"] + node_b["top_views"]
        
        if self.thresholds['top_k_views'] > 0:
            merged_views = sorted(
                merged_views, key=lambda view: view.mask_area, reverse=True,
            )[:self.thresholds['top_k_views']]

        if merged_views:
            descriptors = np.stack([view.descriptor for view in merged_views]).astype(np.float32)
        else:
            raise RuntimeError('this can\'t even really happen so i hope i never get to deal w this ngl')
    # Define a fallback policy.
        best_descriptor, best_idx = self.l1_medoid(descriptors)
        node_a["top_views"] = merged_views
        node_a["descriptor"] = best_descriptor
        node_a['center'] = geom['center']
        node_a['version'] += 1

        self.graph.remove_edges_from(list(self.graph.edges(node_a_id)))
        self.graph.remove_node(node_b_id)
        neighbors.discard(node_a_id)
        neighbors.discard(node_b_id)

        for neighbor_id in neighbors:
            affinity = self._compute_pair_fusion_affinity(
                self.graph.nodes[node_a_id],
                self.graph.nodes[neighbor_id],
            )
            if affinity is not None:
                self._add_edge(node_a_id, neighbor_id, affinity)
        self.graph.remove_node(node_b_id)


    def update_graph(self):
        update_results = dict()
        
        while self.fuse_queue:
            result = self.pop_fusion()
            if result is not None:
                node_a_id, node_b_id = result
                self._fuse_nodes(node_a_id, node_b_id)
        return update_results
