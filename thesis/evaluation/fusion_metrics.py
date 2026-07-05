from dataclasses import dataclass
from itertools import combinations

STRUCTURAL_CLASSES = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
}


@dataclass
class FusionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    precision: float
    recall: float
    f1: float

    evaluable_segments: int
    unique_gt_instances: int
    evaluated_pairs: int

def build_source_to_final_mapping(final_segments) -> dict[int, int]:
    """
    maps each original segment to its final surviving segment
    """
    source_to_final = {}
    for final_segment_id, source_ids in final_segments.items():

        if source_ids is []:
            # A segment that was never fused still represents itself.
            source_ids = {final_segment_id}

        for source_id in source_ids:
            if source_id in source_to_final:
                previous_final_id = source_to_final[source_id]

                raise RuntimeError(
                    f"Source segment {source_id} appears in "
                    f"both final segment {previous_final_id} "
                    f"and final segment {final_segment_id}"
                )
            source_to_final[source_id] = (final_segment_id)

    return source_to_final


def get_evaluable_matches(matches, source_to_final: dict[int, int]):
    """
    keep only reliable, non-structural matches that still have a corresponding final segment
    """
    evaluable_matches = []
    for match in matches:
        if match.status != "matched":
            continue
        if match.gt_class_name in STRUCTURAL_CLASSES:
            continue
        if match.segment_id not in source_to_final:
            continue
        evaluable_matches.append(match)

    return evaluable_matches


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_fusion(matches, final_segments) -> FusionMetrics:
    """
    evaluate pairwise fusion decisions
    for every pair of initial OVO segments:
        same GT instance?
        same final fused segment?
    true_positive: same GT instance and fused together

    false_positive: different GT instances but fused together

    false_negative: same GT instance but still separated

    true_negative: different GT instances and still separated
    """
    source_to_final = build_source_to_final_mapping(final_segments)
    evaluable_matches = get_evaluable_matches(matches=matches, source_to_final=source_to_final)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for first_match, second_match in combinations(evaluable_matches, 2):
        same_gt_instance = first_match.gt_instance_id == second_match.gt_instance_id

        first_final_id = source_to_final[first_match.segment_id]
        second_final_id = source_to_final[second_match.segment_id]

        # did they end up in the same final segment ?
        fused_together = first_final_id == second_final_id

        if same_gt_instance and fused_together:
            true_positives += 1

        elif not same_gt_instance and fused_together:
            false_positives += 1

        elif same_gt_instance and not fused_together:
            false_negatives += 1
        else:
            true_negatives += 1

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    # f1 = safe_divide(int(2 * precision * recall), int(precision + recall))
    f1 = safe_divide(2 * true_positives, 2 * true_positives + false_positives + false_negatives)

    unique_gt_instances = len(
        {
            match.gt_instance_id
            for match in evaluable_matches
        }
    )

    evaluated_pairs = (true_positives + false_positives + false_negatives + true_negatives)

    return FusionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        evaluable_segments=len(evaluable_matches),
        unique_gt_instances=unique_gt_instances,
        evaluated_pairs=evaluated_pairs,
    )
