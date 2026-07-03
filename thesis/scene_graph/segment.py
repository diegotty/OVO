from dataclasses import dataclass, field
from typing import Protocol
import numpy as np

# class SegmentLike(Protocol):
#     id: int
#     descriptor: np.ndarray
#     center: np.ndarray
#     points: np.ndarray
#     keyframe_ids: set[int]
#     version: int

@dataclass
class Segment:
    id: int
    points: np.ndarray
    center: np.ndarray
    descriptor: np.ndarray
    top_views: list[SegmentView]
    # keyframe_ids: set[int]
    # source_segment_ids: set[int] = field(default_factory=set)
    # active: bool = True

    version: int = 1
    persistent: bool = False
    def __init__(self, id : int, points, center, descriptor, top_views : list[SegmentView]):


@dataclass
class SegmentView:
    keyframe_id: int
    descriptor: np.ndarray
    mask_area : float

class SegmentStore:
    def __init__(self, segments : list[Segment]):
        self._segments = {
            segment.id : segment
            for segment in segments
        }
    def get(self, segment_id):
        self._segments[segment_id]


