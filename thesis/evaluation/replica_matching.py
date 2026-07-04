import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from thesis.evaluation.replica_gt import GTInstance

# ai-made, human-proofed
@dataclass
class SegmentMatch:
    """
    result of matching one OVO segment to Replica instances
    """
    segment_id: int
    gt_instance_id: int | None
    gt_class_name: str | None
    best_coverage: float
    second_best_coverage: float
    margin: float
    status: str
    sampled_points: int

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "gt_instance_id": self.gt_instance_id,
            "gt_class_name": self.gt_class_name,
            "best_coverage": self.best_coverage,
            "second_best_coverage": self.second_best_coverage,
            "margin": self.margin,
            "status": self.status,
            "sampled_points": self.sampled_points,
        }


def sample_segment_points(points: np.ndarray, maximum_points: int = 5000) -> np.ndarray:
    """
    limit the number of segment points used for matching.
    this makes matching faster without randomly changing the result between executions.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Segment points must have shape (N, 3), got {points.shape}")

    if len(points) <= maximum_points:
        return points

    # calculates a step and picks maximum_points points (so returns evenly spaced numbers)
    indices = np.linspace(start=0, stop=len(points) - 1, num=maximum_points, dtype=int)
    return points[indices]


def points_inside_instance(points: np.ndarray, instance: GTInstance, tolerance: float = 0.03) -> np.ndarray:
    """
    return one Boolean value for every point:
        t: the point lies inside the Replica oriented bounding box.
        f: the point lies outside it.
    """
    rotation = Rotation.from_quat(instance.rotation_xyzw).as_matrix()

    # Convert points from Replica world coordinates
    # into the local coordinate system of the box.
    local_points = (points - instance.translation) @ rotation
    half_sizes = instance.sizes / 2.0

    lower_bound = ( instance.local_center - half_sizes - tolerance)
    upper_bound = (instance.local_center + half_sizes + tolerance)
    inside = np.all((local_points >= lower_bound) & (local_points <= upper_bound), axis=1)
    return inside


def calculate_instance_coverage(points: np.ndarray, instance: GTInstance, tolerance: float = 0.03) -> float:
    """
    Calculate the fraction of segment points inside one GT box.
    ex: 900 of 1000 points are inside -> coverage = 0.9
    """
    if len(points) == 0:
        return 0.0

    inside = points_inside_instance(points=points, instance=instance, tolerance=tolerance)
    return float(np.mean(inside))


def match_segment_to_instances(
    segment_id: int,
    segment_points: np.ndarray,
    instances: list[GTInstance],
    minimum_coverage: float = 0.50,
    minimum_margin: float = 0.10,
    tolerance: float = 0.03,
    maximum_points: int = 5000,
) -> SegmentMatch:
    """
    match one OVO segment to the Replica instance that contains
    the largest fraction of its points.
    """
    # consider the points of the segment (if too many, a sparser set of them)
    points = sample_segment_points(points=segment_points, maximum_points=maximum_points)

    if len(points) == 0:
        return SegmentMatch(
            segment_id=segment_id,
            gt_instance_id=None,
            gt_class_name=None,
            best_coverage=0.0,
            second_best_coverage=0.0,
            margin=0.0,
            status="empty",
            sampled_points=0,
        )

    scores = []

    for instance in instances:
        coverage = calculate_instance_coverage(points=points, instance=instance, tolerance=tolerance)
        scores.append((instance, coverage))

    scores.sort(key=lambda result: result[1], reverse=True,)
    best_instance, best_coverage = scores[0]

    if len(scores) > 1:
        second_best_coverage = scores[1][1]
    else:
        second_best_coverage = 0.0

    # ok makes sense
    margin = (best_coverage - second_best_coverage)

    if best_coverage < minimum_coverage:
        status = "unmatched"

    elif margin < minimum_margin:
        status = "ambiguous"

    elif best_instance.ignored:
        status = "ignored"
    else:
        status = "matched"

    return SegmentMatch(
        segment_id=segment_id,
        gt_instance_id=best_instance.instance_id,
        gt_class_name=best_instance.class_name,
        best_coverage=best_coverage,
        second_best_coverage=second_best_coverage,
        margin=margin,
        status=status,
        sampled_points=len(points),
    )


def match_segments_to_instances(segments, instances: list[GTInstance]) -> list[SegmentMatch]:
    """
    match every segment in an iterable
    each segment is expected to have: segment.id, segment.points
    """
    matches = []

    for segment in segments:
        match = match_segment_to_instances(
            segment_id=segment.id,
            segment_points=segment.points,
            instances=instances,
        )
        matches.append(match)

    return matches


def save_matches_csv(matches: list[SegmentMatch], output_file: Path) -> None:
    """
    save all matches in a CSV file.
    """
    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # columns of the csv
    field_names = [
        "segment_id",
        "gt_instance_id",
        "gt_class_name",
        "best_coverage",
        "second_best_coverage",
        "margin",
        "status",
        "sampled_points",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for match in matches:
            writer.writerow(match.to_dict())
