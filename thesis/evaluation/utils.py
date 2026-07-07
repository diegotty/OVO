import csv
from pathlib import Path
from _typeshed import ProfileFunction
from collections import Counter, defaultdict
from thesis.evaluation.fusion_metrics import FusionMetrics

def is_evaluable_object_match(match, ignored_classes ) -> bool:
    return (match.status == "matched" and match.gt_class_name not in ignored_classes)

def print_match_summary(matches, ignored_classes) -> None:
    """
    compact summary of the matching result
    """
    status_counts = Counter(match.status for match in matches)
    evaluable_matches = [match for match in matches if is_evaluable_object_match(match, ignored_classes)]
    evaluable_class_counts = Counter(match.gt_class_name for match in evaluable_matches)

    print("---- matching summary ----")
    print(f"total segments: {len(matches)}")

    for status, count in sorted(status_counts.items()):
        print(f"{status:>12}: {count}")

    print(
        f"\nreliable object matches: "
        f"{len(evaluable_matches)}"
    )

    print("\nmatched GT object classes:")
    for class_name, count in evaluable_class_counts.most_common():
        print(f"  {class_name:<20} {count}")

def print_matches(matches):
    for match in matches:
        print(
            f"segment={match.segment_id:<4} "
            f"GT={str(match.gt_instance_id):<4} "
            f"class={str(match.gt_class_name):<15} "
            f"coverage={match.best_coverage:.3f} "
            f"second={match.second_best_coverage:.3f} "
            f"status={match.status}"
        )

def print_gt_fragmentation_summary(matches, ignored_classes) -> None:
    """
    group reliably matched OVO segments by Replica GT instance
    This shows how many initial OVO segments represent each real object.
    """
    segments_by_gt_instance = defaultdict(list)
    class_by_gt_instance = {}
    for match in matches:
        if match.status != "matched":
            continue
        if match.gt_class_name in ignored_classes:
            continue

        gt_instance_id = match.gt_instance_id
        segments_by_gt_instance[gt_instance_id].append(match.segment_id)
        class_by_gt_instance[gt_instance_id] = (match.gt_class_name)

    print("---- GT fragmentation summary ----")
    reliable_object_matches = sum(len(ids) for ids in segments_by_gt_instance.values())]
    unique_gt_instances = len(segments_by_gt_instance
    print(f"reliable matched segments: {reliable_object_matches}")
    print(f"unique GT instances: {unique_gt_instances}")

    # object instance that contains > 1 segments
    fragmented_instances = 0
    
    # number of pairs that need to belong to a single segment 
    positive_segment_pairs = 0
    for gt_instance_id, segment_ids in sorted(segments_by_gt_instance.items()):
            class_name = class_by_gt_instance[gt_instance_id]
            if len(segment_ids) > 1:
                fragmented_instances += 1
    
                # number of unordered, segment pairs that should belong to the GT instance
                # thus all pairs should end up in the same final segment
                # this is useful as we'll use a pairwise metric to check whether all six pairs
                # ultimately belong to the same final segment --- pairwise recall
                positive_segment_pairs += (len(segment_ids)* (len(segment_ids) - 1)// 2)
            print(
                f"GT={gt_instance_id:<4} "
                f"class={class_name:<20} "
                f"segments={len(segment_ids):<3} "
                f"IDs={segment_ids}"
            )
    print(f"fragmented GT instances: {fragmented_instances}")
    print(f"positive fusion pairs: {positive_segment_pairs}")
    return {
        'reliable_object_matches' : reliable_object_matches,
        'unique_gt_instances' : unique_gt_instances,
        'fragmented_gt_instances' : fragmented_instances,
        'positive_same_instance_pairs' : positive_segment_pairs
        'excess_fragments' : reliable_object_matches - unique_gt_instances
    }


def print_fusion_metrics(metrics: FusionMetrics) -> None:
    """
    Print the result of the pairwise fusion evaluation.
    """
    print()
    print("---- Fusion evaluation ----")

    print(f"evaluable segments:   {metrics.evaluable_segments}")
    print(f"unique GT instances:  {metrics.unique_gt_instances}")
    print(f"evaluated pairs:       {metrics.evaluated_pairs}")

    print()
    print("Pair classification:")
    print(f"  true positives:      {metrics.true_positives}")
    print(f"  false positives:     {metrics.false_positives}")
    print(f"  false negatives:     {metrics.false_negatives}")
    print(f"  true negatives:      {metrics.true_negatives}")

    print()
    print("Fusion performance:")
    print(f"  precision:           {metrics.precision:.3f}")
    print(f"  recall:              {metrics.recall:.3f}")
    print(f"  F1 score:            {metrics.f1:.3f}")


def save_stage_summary_csv( first_list: list[dict], second_list: list[dict], third_list: list[dict], output_file: Path) -> None:
    lengths = { len(first_list), len(second_list), len(third_list)}
    if len(lengths) != 1:
        raise ValueError(
            "The three lists must have the same length: "
            f"{len(first_list)}, "
            f"{len(second_list)}, "
            f"{len(third_list)}"
        )

    rows = []
    for index, dictionaries in enumerate( zip( first_list, second_list, third_list)):
        merged_row = {}

        for dictionary in dictionaries:
            for key, value in dictionary.items():
                if ( key in merged_row and merged_row[key] != value):
                    raise ValueError(
                        f"Conflicting value for key '{key}' "
                        f"in row {index}: "
                        f"{merged_row[key]!r} != {value!r}"
                    )
                merged_row[key] = value
        rows.append(merged_row)

    if not rows:
        return

    # Keep columns in their first encountered order.
    field_names = []
    for row in rows:
        for key in row:
            if key not in field_names:
                field_names.append(key)

    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir( parents=True, exist_ok=True)

    with output_file.open( "w", newline="", encoding="utf-8") as file: 
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
