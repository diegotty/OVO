import numpy as np
from shapely.geometry import Polygon

def calculate_iom_poly(obj1, obj2) -> float:
    """
    Calculate intersection over minimum area between the XY projections
    of two axis-aligned 3D bounding boxes.
    """
    min1 = np.asarray(obj1.geometry.bbox_min, dtype=np.float32)
    max1 = np.asarray(obj1.geometry.bbox_max, dtype=np.float32)
    min2 = np.asarray(obj2.geometry.bbox_min, dtype=np.float32)
    max2 = np.asarray(obj2.geometry.bbox_max, dtype=np.float32)

    bbox1 = np.asarray(
        [
            [min1[0], min1[1]],
            [max1[0], min1[1]],
            [max1[0], max1[1]],
            [min1[0], max1[1]],
        ],
        dtype=np.float32,
    )

    bbox2 = np.asarray(
        [
            [min2[0], min2[1]],
            [max2[0], min2[1]],
            [max2[0], max2[1]],
            [min2[0], max2[1]],
        ],
        dtype=np.float32,
    )

    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)

    minimum_area = min(poly1.area, poly2.area)
    if minimum_area <= 0.0:
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    return float(intersection_area / minimum_area)


def relate_above(vertical_iom, on_thres, anchor, target):
    max_z1 = anchor.geometry.bbox_max[2]
    min_z2 = target.geometry.bbox_min[2]
    iom = calculate_iom_poly(anchor, target)

    return max_z1 + on_thres <= min_z2 and iom > vertical_iom

def relate_below(vertical_iom,under_thres, anchor, target):
    max_z_tgt = target.geometry.bbox_max[2]
    min_z_tgt = target.geometry.bbox_min[2]

    max_z_anc = anchor.geometry.bbox_max[2]
    min_z_anc = anchor.geometry.bbox_min[2]
    iom = calculate_iom_poly(anchor, target)

    return max_z_tgt <= min_z_anc and iom > vertical_iom or min_z_tgt <= min_z_anc + under_thres and max_z_tgt <= max_z_anc and iom > vertical_iom


def get_aabb_distance(segment_a, segment_b) -> float:
    """
    Return the shortest Euclidean distance between two AABBs.

    The distance is:
    - zero if the boxes touch or intersect;
    - positive if they are separated.
    """
    min_a = np.asarray(segment_a.geometry.bbox_min, dtype=np.float32)
    max_a = np.asarray(segment_a.geometry.bbox_max, dtype=np.float32)
    min_b = np.asarray(segment_b.geometry.bbox_min, dtype=np.float32)
    max_b = np.asarray(segment_b.geometry.bbox_max, dtype=np.float32)
    axis_gaps = np.maximum(np.maximum(min_b - max_a,min_a - max_b), 0.0)
    return float(np.linalg.norm(axis_gaps))

def relate_near(near_threshold, anchor, target):
    distance = get_aabb_distance(anchor, target)
    return distance <= near_threshold

def compute_spatial_relations(anchor, segment_store, spatial_relations, thresholds):
    vertical_iom = thresholds["vertical_iom"]
    near_thres = thresholds["near_thres"]
    on_thres = thresholds["on_thres"]
    under_thres = thresholds["under_thres"]

    relations = {
        relation : []
        for relation in spatial_relations
    }

    for target in segment_store.segments(active_only=True, persistent_only=True):
        if target.id == anchor.id:
            continue
        if relate_above(vertical_iom, on_thres, anchor, target) and 'above' in relations:
            relations['above'].append(target.id)
        if relate_below(vertical_iom, under_thres, anchor, target) and 'below' in relations:
            relations['below'].append(target.id)
        if relate_near(near_thres, anchor, target) and 'near' in relations:
            relations['near'].append(target.id)
    return relations
