from collections import Counter

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
