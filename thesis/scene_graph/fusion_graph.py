import networkx as nx
import open3d as o3d
import numpy as np
from thesis.scene_graph.utils import graph_utils
from itertools import combinations
import heapq

class FusionGraph:
    '''
    builds an graph with segments and nodes, and edges that define the combined similarity
    between a pair of nodes. edges are only created if the affinity exceeds affinity_threshold
    '''
    def __init__(self, segment_store, thresholds, debug=False):
        # TODO: will hae to make this modular 
        self.thresholds = thresholds
        self.fuse_queue = []
        self.debug = debug
        self.graph = nx.Graph()
        self.segment_store = segment_store
        for segment in segment_store.segments(not_absorbed_only=True):
            self.graph.add_node(segment.id)
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
    def _compute_pair_fusion_affinity(self, node_a_id, node_b_id):
        node_a = self.segment_store.get(node_a_id)
        node_b = self.segment_store.get(node_b_id)
        distance, proximity = graph_utils.geometric_affinity(
            node_a.geometry.center(), 
            node_b.geometry.center(), 
            self.thresholds.get('distance_scale', 1)
        )
        # if distance > self.thresholds['distance_threshold']:
        #     return None

        cosine, semantic_similarity = graph_utils.semantic_similarity(
            node_a.descriptor, 
            node_b.descriptor)
        # if semantic_similarity < self.thresholds['semantic_threshold']:
        #    return None
    
        coverage_ab, coverage_ba = self.point_coverage(
            node_a.points, 
            node_b.points, 
            self.thresholds['point_distance_threshold']
        )
        max_coverage = max(coverage_ab, coverage_ba)
        # min_coverage = min(coverage_ab, coverage_ba)

        proximity_score = np.clip((self.thresholds['distance_threshold'] - distance) / ( self.thresholds['distance_threshold']), 0.0, 1.0)
        semantic_score = np.clip((semantic_similarity - self.thresholds['semantic_threshold']) / (1.0 - self.thresholds['semantic_threshold']), 0.0, 1.0)

        # handling the 2 cases
        required_coverage = self.thresholds['weak_coverage_threshold'] if semantic_similarity > self.thresholds['strong_semantic_threshold'] else self.thresholds['coverage_threshold']
        coverage_score = np.clip((max_coverage - required_coverage) / (1.0 - required_coverage), 0.0, 1.0)

        # TODO, FIX: use good condition
        can_fuse = (
            max_coverage > self.thresholds['coverage_threshold']
            or (semantic_similarity > self.thresholds['strong_semantic_threshold']
                and max_coverage > self.thresholds['weak_coverage_threshold'])
        )
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
        # print(f'added edge: ({node_a_id},{node_b_id}, with fuse score:  {affinity["fuse_score"]}')
        self.graph.add_edge(node_a_id, node_b_id, 
            # centroid_distance = affinity['centroid_distance'],
            # semantic_affinity = affinity['semantic_affinity'],
            # pcd_coverage = affinity['pcd_coverage'],
            fuse_score = affinity['fuse_score'],
            can_fuse = affinity['can_fuse']
        )
        #if affinity['can_fuse']:
        if affinity['fuse_score'] > 0.5:
            version_a = self.segment_store.get(node_a_id).version
            version_b = self.segment_store.get(node_b_id).version
            # print(f'added to queue: ({node_a_id}, {node_b_id}) with fuse score {affinity["fuse_score"]}')
            heapq.heappush(
                self.fuse_queue, 
                (-affinity['fuse_score'],
                 node_a_id, version_a,
                 node_b_id, version_b)
            )

    def _init_edges(self):
        node_ids = list(self.graph.nodes)
        for node_a_id, node_b_id in combinations(node_ids, 2):
            affinity = self._compute_pair_fusion_affinity(node_a_id, node_b_id)
            # distance_scale, dist_th, point_dist_th, semantic_th, coverage_th)
            if affinity is not None:
                self._add_edge(node_a_id, node_b_id, affinity)

    def pop_fusion(self):
        while self.fuse_queue:
            (neg_fuse_score,
             node_a_id, q_version_a,
             node_b_id, q_version_b) = heapq.heappop(self.fuse_queue) 

            # nodes not in graph
            if node_a_id not in self.graph or node_b_id not in self.graph:
                continue
            # edge got removed
            if not self.graph.has_edge(node_a_id, node_b_id):
                continue
            # wrong version
            if self.segment_store.get(node_a_id).version != q_version_a or \
               self.segment_store.get(node_b_id).version != q_version_b: 
                continue

            edge = self.graph.edges[node_a_id, node_b_id]
            if edge['fuse_score'] < 0.5:
            #if not edge["can_fuse"] or edge['fuse_score'] < 0.5:
                continue

            print(f'fusing {node_b_id} into {node_a_id}, with fuse_score={-neg_fuse_score}')
            return node_a_id, node_b_id
        # no nodes to fuse
        return None


    def _fuse_nodes(self, node_a_id, node_b_id):
        # TODO: for real-time implementation, add keyframes and update the heap of survivor
        self.segment_store.fuse(node_a_id, node_b_id, self.thresholds['top_k_views'])
        neighbors_a = set(self.graph.neighbors(node_a_id))
        neighbors_a.discard(node_a_id)
        self.graph.remove_edges_from(list(self.graph.edges(node_a_id)))

        neighbors_b = set(self.graph.neighbors(node_b_id))
        self.graph.remove_node(node_b_id)

        neighbors = neighbors_a.union(neighbors_b)
        neighbors.discard(node_a_id)
        neighbors.discard(node_b_id)

        # print(f'recalculating edges for node {node_a_id}')
        for neighbor_id in neighbors:
            affinity = self._compute_pair_fusion_affinity(node_a_id, neighbor_id)
            if affinity is not None:
                self._add_edge(node_a_id, neighbor_id, affinity)

    def update_graph(self):
        map = dict()
        parts = {
            'absorbed' : set(),
            'survivors' : set()
        }
        while self.fuse_queue:
            result = self.pop_fusion()
            if result is not None:
                node_a_id, node_b_id = result
                self._fuse_nodes(node_a_id, node_b_id)
                # fused: survivor
                parts['absorbed'].add(node_b_id)
                parts['survivors'].discard(node_b_id)
                parts['survivors'].add(node_a_id)
                map[node_b_id] = node_a_id
        return map, parts
