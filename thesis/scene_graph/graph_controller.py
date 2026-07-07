from pathlib import Path
from thesis.scene_graph.spatial_graph import SpatialGraph
from thesis.scene_graph.fusion_graph import FusionGraph
from thesis.scene_graph.segment import SegmentStore, SegmentState
from thesis.scene_graph.utils import load_utils

# OVO/thesis/scene_graph/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

class Controller:
    fusion_graph : FusionGraph
    spatial_graph : SpatialGraph
    segment_store : SegmentStore
    config : dict

    def __init__(self, export_dir):
        config_path = SCRIPT_DIR / "config.yaml"
        self.config = load_utils.load_config(config_path)

        self.segment_store = load_utils.load_segments(export_dir, self.config['min_segment_points'])
        fusion_thresholds = self.config['fusion'].copy()
        fusion_thresholds['top_k_views'] = self.config['top_k_views']
        self.fusion_graph = FusionGraph(self.segment_store, fusion_thresholds)
    
        spatial_thresholds = self.config['spatial_graph'].copy()
        spatial_relations = []
        for relation, value in self.config['spatial_graph']['relations'].items():
            if value:
                spatial_relations.append(relation)
    
        self.spatial_graph = SpatialGraph(self.segment_store, spatial_relations, spatial_thresholds)

    def persistence_filter(self):
        for segment in self.segment_store.segments():
            if segment.state is not SegmentState.ABSORBED:
                if len(segment.keyframe_ids) < self.config['persistence_threshold']:
                    segment.state = SegmentState.TENTATIVE
 

    def update_graphs(self):
        updates = self.fusion_graph.update_graph()
        self.persistence_filter()
        self.spatial_graph.update_graph(updates)
        return updates
