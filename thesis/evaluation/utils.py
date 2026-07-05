from collections import Counter, defaultdict

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
    print(
        f"reliable matched segments: {sum(len(ids) for ids in segments_by_gt_instance.values())}")
    print(f"unique GT instances: {len(segments_by_gt_instance)}")

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
