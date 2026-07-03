from typing import Dict, Any
import yaml
import json
from pathlib import Path
from pprint import pprint
from typing import Any
import numpy as np
import networkx as nx

# ai-made
def print_region_summary(region: dict) -> None:
    objects = region["objects"]

    print("\n=== REGION SUMMARY ===", flush=True)
    print("region_id:", region["region_id"], flush=True)
    print("region_name:", region["region_name"], flush=True)
    print("number of objects:", len(objects), flush=True)

    region_bbox = np.asarray(region["region_bbox"])
    print("region_bbox shape:", region_bbox.shape, flush=True)
    print("region_bbox:", flush=True)
    print(region_bbox, flush=True)

    print("\n=== OBJECTS ===", flush=True)

    for obj in objects:
        print(
            f"object_id={obj['object_id']} "
            f"center={np.asarray(obj['center'])} "
            f"size={np.asarray(obj['size'])} "
            f"volume={obj['volume']:.6f} "
            f"bbox_shape={np.asarray(obj['bbox']).shape} "
            f"nyu_id={obj.get('nyu_id')}",
            flush=True,
        )

    print("======================\n", flush=True)

def print_relations(region):
    for relation, anchors in region["relationships"].items():
        print(f"\n=== {relation.upper()} ===")

        for anchor_id, targets in anchors.items():
            if not targets:
                continue

            for target in targets:
                if relation == "between":
                    print(f"{anchor_id} {relation} {target[0]} and {target[1]}" )
                else:
                    print(f"{target} {relation} {anchor_id}")

def print_relation_graph_summary(
    graph: nx.MultiDiGraph,
) -> None:
    """
    Print basic graph and relation statistics.
    """

    relation_counts = Counter(
        edge_data["relation"]
        for _, _, edge_data in graph.edges(data=True)
    )

    print("\n=== FULL RELATION GRAPH ===")
    print("nodes:", graph.number_of_nodes())
    print("edges:", graph.number_of_edges())
    print("affinity filtered:", graph.graph["affinity_filtered"])

    print("\nRelations:")

    for relation_name in graph.graph["relation_set"]:
        print(
            f"{relation_name}: "
            f"{relation_counts.get(relation_name, 0)}"
        )

def print_relation_graph_edges(
    graph: nx.MultiDiGraph,
) -> None:
    print("\n=== RELATION EDGES ===")

    for source, target, key, data in graph.edges(
        keys=True,
        data=True,
    ):
        print(
            f"{source} --[{data['relation']}]--> {target}"
        )
def validate_relation_graph(
    graph: nx.MultiDiGraph,
) -> None:
    """
    Validate expected inverse and symmetric relations.
    """

    errors = []

    for source, target, key, data in graph.edges(
        keys=True,
        data=True,
    ):
        relation = data["relation"]

        if relation == "above":
            if not graph.has_edge(
                target,
                source,
                key="below",
            ):
                errors.append(
                    f"{source} above {target}, "
                    f"but {target} below {source} is missing"
                )

        elif relation == "below":
            if not graph.has_edge(
                target,
                source,
                key="above",
            ):
                errors.append(
                    f"{source} below {target}, "
                    f"but {target} above {source} is missing"
                )

        elif relation == "near":
            if not graph.has_edge(
                target,
                source,
                key="near",
            ):
                errors.append(
                    f"{source} near {target}, "
                    f"but reverse near edge is missing"
                )

    if errors:
        print("\n=== RELATION VALIDATION ERRORS ===")

        for error in errors[:20]:
            print(error)

        print("total errors:", len(errors))

    else:
        print(
            "\nRelation graph validation passed."
        )
