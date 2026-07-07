from dataclasses import dataclass, field
from thesis.scene_graph.spatial_relations.geometry import compute_aabb, SegmentGeometry
from typing import Protocol
from collections.abc import Iterator
import numpy as np
from enum import Enum, auto

# class SegmentLike(Protocol):
#     id: int
#     descriptor: np.ndarray
#     center: np.ndarray
#     points: np.ndarray
#     keyframe_ids: set[int]
#     version: int

class SegmentState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    ABSORBED = auto()       


@dataclass
class SegmentView:
    keyframe_id: int
    descriptor: np.ndarray
    mask_area : float

# gpt-made
def l1_medoid(descriptors):
    descriptors = np.asarray(descriptors, dtype=np.float32)
    pairwise_distances = np.abs(descriptors[:, None, :] - descriptors[None, :, :]).sum(axis=2)
    best_idx = int(np.argmin(pairwise_distances.sum(axis=1)))
    return descriptors[best_idx].copy(), best_idx

@dataclass
class Segment:
    id: int
    points: np.ndarray
    descriptor: np.ndarray
    top_views: list[SegmentView]
    geometry : SegmentGeometry
    keyframe_ids: set[int]
    version: int = 1
    # we think of persitance as monotonic (as observations only increase)
    state : SegmentState = SegmentState.CONFIRMED

class SegmentStore:
    def __init__(self, segments : list[Segment]):
        self._segments = {
            segment.id : segment
            for segment in segments
        }
    def get(self, segment_id):
        return self._segments[segment_id]


    def fuse(self, node_a_id : int, node_b_id : int, top_k_views : int):
        node_a = self._segments[node_a_id]
        node_b = self._segments[node_b_id]
        node_b.state = SegmentState.ABSORBED
        # node_a["points"] = np.concatenate([node_a["points"], node_b["points"]], axis=0)
        node_a.points = np.concatenate((node_a.points, node_b.points), axis=0)
        node_a.geometry = compute_aabb(node_a.id, node_a.points)

        merged_views = node_a.top_views + node_b.top_views
        if top_k_views > 0:
            merged_views = sorted(
                merged_views, key=lambda view: view.mask_area, reverse=True,
            )[:top_k_views]

        if merged_views:
            descriptors = np.stack([view.descriptor for view in merged_views]).astype(np.float32)
        else:
            raise RuntimeError('this can\'t even really happen so i hope i never get to deal w this ngl')

        best_descriptor, best_idx = l1_medoid(descriptors)
        node_a.top_views = merged_views
        node_a.descriptor = best_descriptor
        node_a.version += 1

    def segments(self, confirmed_only: bool = False, not_absorbed_only: bool = False) -> Iterator[Segment]:
        for segment in self._segments.values():
            if confirmed_only and not segment.state == SegmentState.CONFIRMED:
                continue
            if not_absorbed_only and segment.state == SegmentState.ABSORBED:
                continue
            yield segment
