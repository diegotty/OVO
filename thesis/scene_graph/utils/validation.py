from dataclasses import dataclass, field
from typing import Dict, Any
from thesis.scene_graph.segment import Segment, SegmentGeometry, SegmentStore
from thesis.scene_graph.fusion_graph import FusionGraph
from thesis.scene_graph.spatial_graph import SpatialGraph
import yaml
import json
from pathlib import Path
from pprint import pprint
from typing import Any
import numpy as np
import networkx as nx
from collections import Counter

# ai-made, human-proofed
import numpy as np

@dataclass
class Validation:
    flags : dict
    segment_store : SegmentStore
    fusion_graph : FusionGraph
    spatial_graph : SpatialGraph
    initial_active_count : int
    fusion_updates: dict[Any, Any] = field(default_factory=dict)


    def validate_segment_store(self, stage: str) -> None:
        segments = list(self.segment_store.segments())
        active_segments = list(self.segment_store.segments(not_absorbed_only=True))
        confirmed_segments = list(self.segment_store.segments(confirmed_only=True))
        segment_ids = [segment.id for segment in segments]
        active_ids = [segment.id for segment in active_segments]
    
        print(f"\n--- segment_store: {stage} ---")
        print(f"total segments:  {len(segments)}")
        print(f"active segments: {len(active_segments)}")
        print(f"confirmed segments: {len(confirmed_segments)}")
        print(f"inactive:        {len(segments) - len(active_segments)}")
    
        assert len(segment_ids) == len(set(segment_ids)), (
            "segment_store contains duplicate segment IDs"
        )
    
        descriptor_shapes = {
            segment.descriptor.shape
            for segment in active_segments
        }
    
        point_counts = [
            len(segment.points)
            for segment in active_segments
        ]
    
        print(f"descriptor shapes: {descriptor_shapes}")
    
        if point_counts:
            print(
                "points per segment: "
                f"min={min(point_counts)}, "
                f"mean={np.mean(point_counts):.1f}, "
                f"max={max(point_counts)}"
            )
    
        for segment in active_segments:
            assert segment.points.ndim == 2
            assert segment.points.shape[1] == 3
    
            assert np.all(np.isfinite(segment.points)), (
                f"Segment {segment.id} has invalid points"
            )
    
            assert np.all(np.isfinite(segment.descriptor)), (
                f"Segment {segment.id} has invalid descriptor"
            )
    
            assert np.all(
                segment.geometry.aabb_min
                <= segment.geometry.aabb_min
            ), f"Invalid AABB for segment {segment.id}"
    
            assert np.all(
                np.isfinite(segment.geometry.center())
            ), f"Invalid center for segment {segment.id}"
    
        print(f"active IDs sample: {active_ids[:10]}")

    def validate_fusion_graph(self, stage: str) -> None:
        graph = self.fusion_graph.graph
        segment_store = self.fusion_graph.segment_store
    
        active_ids = {
            segment.id
            for segment in segment_store.segments(not_absorbed_only=True)
        }
    
        graph_ids = set(graph.nodes)
        edges = list(graph.edges(data=True))
        fusion_candidates = [
            (node_a, node_b, data)
            for node_a, node_b, data in edges
            if data.get("can_fuse", True)
        ]
    
        print(f"\n--- fusion_graph: {stage} ---")
        print(f"nodes:             {graph.number_of_nodes()}")
        print(f"edges:             {graph.number_of_edges()}")
        print(f"fusion candidates: {len(fusion_candidates)}")
        print(f"heap entries:      {len(self.fusion_graph.fuse_queue)}")
    
        assert graph_ids == active_ids, (
            "fusion_graph nodes do not match active segment_store IDs\n"
            f"missing from graph: {active_ids - graph_ids}\n"
            f"unknown graph IDs:  {graph_ids - active_ids}"
        )
    
        for node_a, node_b, data in edges:
            score = float(data["fuse_score"])
            assert 0.0 <= score <= 1.0, (
                f"Invalid fusion score for ({node_a}, {node_b}): "
                f"{score}"
            )
    
        top_candidates = sorted(
            fusion_candidates,
            key=lambda edge: edge[2]["fuse_score"],
            reverse=True,
        )[:10]
    
        print("top fusion candidates:")
        for node_a, node_b, data in top_candidates:
            print(
                f"  {node_a} <-> {node_b}: "
                f"{data['fuse_score']:.4f}"
            )
    def validate_fusion_updates(self, initial_active_count: int) -> None:
        absorbed = set(self.fusion_updates.get("absorbed", set()))
        survivors = set(self.fusion_updates.get("survivors", set()))
    
        active_ids = {
            segment.id
            for segment in self.segment_store.segments(not_absorbed_only=True)
        }
    
        print("\n--- fusion results ---")
        print(f"survivors modified: {len(survivors)}")
        print(f"segments absorbed:  {len(absorbed)}")
        print(f"survivor IDs:       {sorted(survivors)}")
        print(f"absorbed IDs:       {sorted(absorbed)}")
        print(
            f"active segments:    "
            f"{initial_active_count} -> {len(active_ids)}"
        )
    
        assert absorbed.isdisjoint(active_ids), (
            "Absorbed segments are still active"
        )
    
        assert survivors <= active_ids, (
            "Some survivors are no longer active"
        )
    
        assert absorbed.isdisjoint(self.fusion_graph.graph.nodes), (
            "Absorbed segments remain in FusionGraph"
        )
    
        expected_count = initial_active_count - len(absorbed)
        assert len(active_ids) == expected_count, (
            f"Expected {expected_count} active segments, "
            f"found {len(active_ids)}"
        )



    def validate_spatial_graph(self, stage : str) -> None:
        graph = self.spatial_graph.graph
        segment_store = self.spatial_graph.segment_store
    
        active_ids = {
            segment.id
            for segment in segment_store.segments(not_absorbed_only=True)
        }
    
        graph_ids = set(graph.nodes)
        print("\n--- spatial_graph: {stage} ---")
        print(f"nodes: {graph.number_of_nodes()}")
        print(f"edges: {graph.number_of_edges()}")
    
        assert graph_ids == active_ids, (
            "spatial_graph nodes do not match active SegmentStore IDs\n"
            f"missing from graph: {active_ids - graph_ids}\n"
            f"unknown graph IDs:  {graph_ids - active_ids}"
        )
    
        relation_counts = Counter()
        relation_triples = set()
    
        for source, target, key, data in graph.edges(keys=True, data=True):
            relation = data.get("relation", key)
            relation_counts[relation] += 1
            relation_triples.add((source, target, relation))
    
            assert source in active_ids
            assert target in active_ids
            assert source != target, (
                f"self-relation detected: "
                f"{source} --{relation}--> {target}"
            )
    
        assert len(relation_triples) == graph.number_of_edges(), (
            "Duplicate spatial relation edges detected"
        )
    
        print("relations count:")
        for relation, count in sorted(
            relation_counts.items()
        ):
            print(f"  {relation}: {count}")
    
        print("sample edges:")
        for source, target, key, data in list(graph.edges(keys=True, data=True))[:15]:
            relation = data.get("relation", key)
    
            print(
                f"  {source} --{relation}--> {target}"
                "  [target segment -> anchor segment]"
            )

    def validate(self, phase : str):
        if self.flags['segment_store']:
            self.validate_segment_store(phase)
        if self.flags['fusion_graph']:
            self.validate_fusion_graph(phase)
        if self.flags['fusion_updates']:
            self.validate_fusion_updates(self.initial_active_count)
        if self.flags['spatial_graph']:
            self.validate_spatial_graph(phase)
