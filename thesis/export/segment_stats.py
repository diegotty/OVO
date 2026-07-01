#!/usr/bin/env python3
"""
Compute persistence and point-support statistics from exported OVO scenes.

The script accepts either:
- an export directory containing scene.json, or
- the scene.json file itself.

Examples
--------
Single scene:
    python segment_stats.py path/to/export_dir

Multiple scenes:
    python segment_stats.py \
        path/to/scene_1/scene.json \
        path/to/scene_2/scene.json

Custom K_min values:
    python segment_stats.py path/to/export_dir --kmins 1 3 5 7 10

Outputs
-------
segment_stats_output/
    segment_stats.csv
    stats_by_observation_count.csv
    stats_by_observation_bracket.csv
    correlations.csv
    kmin_retention.csv
    segment_count_by_observations.png
    points_vs_observations.png
    mean_points_by_observation_bracket.png
    kmin_retention.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


# Brackets are centred around the tentative K_min = 7 decision.
# Edit these values if you want a different grouping.
OBSERVATION_BRACKETS: tuple[tuple[int, int | None], ...] = (
    (1, 1),
    (2, 2),
    (3, 4),
    (5, 6),
    (7, 9),
    (10, None),
)


@dataclass(frozen=True)
class SegmentStat:
    scene: str
    scene_json: str
    segment_id: int
    n_observations: int
    n_points: int

    @property
    def points_per_observation(self) -> float:
        if self.n_observations == 0:
            return float("nan")
        return self.n_points / self.n_observations


def resolve_scene_json(path: Path) -> Path:
    """Resolve either an export directory or a scene.json path."""
    path = path.expanduser().resolve()

    if path.is_dir():
        scene_json = path / "scene.json"
    else:
        scene_json = path

    if not scene_json.is_file():
        raise FileNotFoundError(
            f"Could not find scene.json at: {scene_json}"
        )

    return scene_json


def load_scene(scene_json: Path) -> list[SegmentStat]:
    """Load one exported scene and compute per-segment statistics."""
    with scene_json.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if "segments" not in metadata:
        raise KeyError(f"{scene_json} does not contain a 'segments' field.")

    scene_name = scene_json.parent.name
    results: list[SegmentStat] = []
    seen_ids: set[int] = set()

    for segment in metadata["segments"]:
        required = {"id", "points_file", "keyframe_ids"}
        missing = required.difference(segment)

        if missing:
            raise KeyError(
                f"{scene_json}: segment entry is missing {sorted(missing)}"
            )

        segment_id = int(segment["id"])

        if segment_id in seen_ids:
            raise ValueError(
                f"{scene_json}: duplicate segment ID {segment_id}"
            )
        seen_ids.add(segment_id)

        # OVO normally stores unique keyframe IDs, but set() is used
        # defensively in case an exported file contains duplicates.
        keyframe_ids = {
            int(keyframe_id)
            for keyframe_id in segment["keyframe_ids"]
        }
        n_observations = len(keyframe_ids)

        points_path = scene_json.parent / segment["points_file"]
        if not points_path.is_file():
            raise FileNotFoundError(
                f"Segment {segment_id}: missing point file {points_path}"
            )

        # mmap_mode reads only the array header and shape, avoiding loading
        # every complete point cloud into memory.
        points = np.load(points_path, mmap_mode="r")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Segment {segment_id}: expected an N x 3 point array, "
                f"got shape {points.shape}"
            )

        results.append(
            SegmentStat(
                scene=scene_name,
                scene_json=str(scene_json),
                segment_id=segment_id,
                n_observations=n_observations,
                n_points=int(points.shape[0]),
            )
        )

    return results


def observation_bracket(n_observations: int) -> str:
    if n_observations <= 0:
        return "0"

    for lower, upper in OBSERVATION_BRACKETS:
        if upper is None and n_observations >= lower:
            return f"{lower}+"

        if upper is not None and lower <= n_observations <= upper:
            return str(lower) if lower == upper else f"{lower}-{upper}"

    raise ValueError(
        f"No bracket configured for {n_observations} observations."
    )


def bracket_sort_key(label: str) -> int:
    if label == "0":
        return 0
    return int(label.split("-")[0].removesuffix("+"))


def summarize_point_counts(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)

    if array.size == 0:
        raise ValueError("Cannot summarize an empty collection.")

    return {
        "n_segments": int(array.size),
        "mean_points": float(array.mean()),
        "median_points": float(np.median(array)),
        "std_points": float(array.std(ddof=0)),
        "min_points": int(array.min()),
        "max_points": int(array.max()),
        "total_points": int(array.sum()),
    }


def group_statistics(
    stats: Sequence[SegmentStat],
    group_function: Callable[[SegmentStat], int | str],
    group_name: str,
    sort_function: Callable[[int | str], object],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int | str], list[int]] = defaultdict(list)

    for stat in stats:
        group_value = group_function(stat)
        grouped[(stat.scene, group_value)].append(stat.n_points)
        grouped[("ALL", group_value)].append(stat.n_points)

    rows: list[dict[str, object]] = []

    for (scene, group_value), point_counts in grouped.items():
        rows.append(
            {
                "scene": scene,
                group_name: group_value,
                **summarize_point_counts(point_counts),
            }
        )

    rows.sort(
        key=lambda row: (
            row["scene"] == "ALL",
            str(row["scene"]),
            sort_function(row[group_name]),
        )
    )

    return rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for tied values, used for Spearman correlation."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calculate_correlations(
    stats: Sequence[SegmentStat],
) -> list[dict[str, object]]:
    by_scene: dict[str, list[SegmentStat]] = defaultdict(list)

    for stat in stats:
        by_scene[stat.scene].append(stat)
        by_scene["ALL"].append(stat)

    rows: list[dict[str, object]] = []

    for scene, scene_stats in sorted(
        by_scene.items(),
        key=lambda item: (item[0] == "ALL", item[0]),
    ):
        observations = np.asarray(
            [stat.n_observations for stat in scene_stats],
            dtype=np.float64,
        )
        points = np.asarray(
            [stat.n_points for stat in scene_stats],
            dtype=np.float64,
        )

        rows.append(
            {
                "scene": scene,
                "n_segments": len(scene_stats),
                "pearson_observations_points": safe_pearson(
                    observations, points
                ),
                "spearman_observations_points": safe_pearson(
                    average_ranks(observations),
                    average_ranks(points),
                ),
            }
        )

    return rows


def calculate_kmin_retention(
    stats: Sequence[SegmentStat],
    kmins: Sequence[int],
) -> list[dict[str, object]]:
    """
    For each K_min, report retained segments and retained point support.

    Point retention is not object recall. It only indicates how much of the
    segment-labelled point cloud remains represented by admitted nodes.
    """
    by_scene: dict[str, list[SegmentStat]] = defaultdict(list)

    for stat in stats:
        by_scene[stat.scene].append(stat)
        by_scene["ALL"].append(stat)

    rows: list[dict[str, object]] = []

    for scene, scene_stats in sorted(
        by_scene.items(),
        key=lambda item: (item[0] == "ALL", item[0]),
    ):
        total_segments = len(scene_stats)
        total_points = sum(stat.n_points for stat in scene_stats)

        for kmin in sorted(set(kmins)):
            retained = [
                stat
                for stat in scene_stats
                if stat.n_observations >= kmin
            ]
            retained_points = sum(stat.n_points for stat in retained)

            rows.append(
                {
                    "scene": scene,
                    "k_min": kmin,
                    "total_segments": total_segments,
                    "retained_segments": len(retained),
                    "segment_retention_ratio": (
                        len(retained) / total_segments
                        if total_segments > 0
                        else float("nan")
                    ),
                    "total_points": total_points,
                    "retained_points": retained_points,
                    "point_retention_ratio": (
                        retained_points / total_points
                        if total_points > 0
                        else float("nan")
                    ),
                }
            )

    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def print_table(
    title: str,
    rows: Sequence[dict[str, object]],
    columns: Sequence[str],
) -> None:
    print(f"\n{title}")

    if not rows:
        print("No data.")
        return

    formatted = [
        [format_value(row[column]) for column in columns]
        for row in rows
    ]

    widths = [
        max(
            len(column),
            max(len(row[index]) for row in formatted),
        )
        for index, column in enumerate(columns)
    ]

    print(
        " | ".join(
            column.ljust(widths[index])
            for index, column in enumerate(columns)
        )
    )
    print("-+-".join("-" * width for width in widths))

    for row in formatted:
        print(
            " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def create_plots(
    stats: Sequence[SegmentStat],
    exact_rows: Sequence[dict[str, object]],
    bracket_rows: Sequence[dict[str, object]],
    retention_rows: Sequence[dict[str, object]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nMatplotlib is not installed. CSV files were generated, "
            "but plots were skipped."
        )
        return

    combined_exact = [
        row for row in exact_rows if row["scene"] == "ALL"
    ]

    x_obs = [int(row["n_observations"]) for row in combined_exact]
    y_segments = [int(row["n_segments"]) for row in combined_exact]

    plt.figure(figsize=(10, 5))
    plt.bar(x_obs, y_segments)
    plt.xlabel("Number of semantic observations")
    plt.ylabel("Number of 3D segments")
    plt.title("3D segments grouped by observation count")
    plt.tight_layout()
    plt.savefig(
        output_dir / "segment_count_by_observations.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [stat.n_observations for stat in stats],
        [stat.n_points for stat in stats],
        alpha=0.65,
    )
    plt.xlabel("Number of semantic observations")
    plt.ylabel("Number of map points")
    plt.title("Point support versus observation count")
    plt.tight_layout()
    plt.savefig(
        output_dir / "points_vs_observations.png",
        dpi=200,
    )
    plt.close()

    combined_brackets = [
        row for row in bracket_rows if row["scene"] == "ALL"
    ]
    combined_brackets.sort(
        key=lambda row: bracket_sort_key(
            str(row["observation_bracket"])
        )
    )

    plt.figure(figsize=(9, 5))
    plt.bar(
        [
            str(row["observation_bracket"])
            for row in combined_brackets
        ],
        [
            float(row["mean_points"])
            for row in combined_brackets
        ],
    )
    plt.xlabel("Semantic observation bracket")
    plt.ylabel("Average number of map points")
    plt.title("Average point support by observation bracket")
    plt.tight_layout()
    plt.savefig(
        output_dir / "mean_points_by_observation_bracket.png",
        dpi=200,
    )
    plt.close()

    combined_retention = [
        row for row in retention_rows if row["scene"] == "ALL"
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(
        [int(row["k_min"]) for row in combined_retention],
        [
            float(row["segment_retention_ratio"]) * 100
            for row in combined_retention
        ],
        marker="o",
        label="Segments retained",
    )
    plt.plot(
        [int(row["k_min"]) for row in combined_retention],
        [
            float(row["point_retention_ratio"]) * 100
            for row in combined_retention
        ],
        marker="o",
        label="Points retained",
    )
    plt.xlabel("K_min")
    plt.ylabel("Retention (%)")
    plt.title("Effect of persistence threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "kmin_retention.png",
        dpi=200,
    )
    plt.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute persistence and point-support statistics from "
            "exported OVO scenes."
        )
    )

    parser.add_argument(
        "scenes",
        nargs="+",
        type=Path,
        help=(
            "One or more export directories or paths to scene.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segment_stats_output"),
        help="Directory for CSV files and plots.",
    )

    parser.add_argument(
        "--kmins",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 10],
        help="Persistence thresholds used in the retention analysis.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if any(kmin < 0 for kmin in args.kmins):
        raise ValueError("K_min values must be non-negative.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[SegmentStat] = []

    for input_path in args.scenes:
        scene_json = resolve_scene_json(input_path)
        scene_stats = load_scene(scene_json)
        all_stats.extend(scene_stats)

        print(
            f"Loaded {len(scene_stats)} segments from {scene_json}"
        )

    if not all_stats:
        raise RuntimeError("No segment statistics were extracted.")

    segment_rows = [
        {
            "scene": stat.scene,
            "scene_json": stat.scene_json,
            "segment_id": stat.segment_id,
            "n_keyframe_observations": stat.n_observations,
            "observation_bracket": observation_bracket(
                stat.n_observations
            ),
            "n_points": stat.n_points,
            "points_per_observation": stat.points_per_observation,
        }
        for stat in sorted(
            all_stats,
            key=lambda item: (
                item.scene,
                item.n_observations,
                item.segment_id,
            ),
        )
    ]

    exact_rows = group_statistics(
        all_stats,
        group_function=lambda stat: stat.n_observations,
        group_name="n_observations",
        sort_function=lambda value: int(value),
    )

    bracket_rows = group_statistics(
        all_stats,
        group_function=lambda stat: observation_bracket(
            stat.n_observations
        ),
        group_name="observation_bracket",
        sort_function=lambda value: bracket_sort_key(str(value)),
    )

    correlation_rows = calculate_correlations(all_stats)
    retention_rows = calculate_kmin_retention(
        all_stats,
        args.kmins,
    )

    write_csv(output_dir / "segment_stats.csv", segment_rows)
    write_csv(
        output_dir / "stats_by_observation_count.csv",
        exact_rows,
    )
    write_csv(
        output_dir / "stats_by_observation_bracket.csv",
        bracket_rows,
    )
    write_csv(output_dir / "correlations.csv", correlation_rows)
    write_csv(output_dir / "kmin_retention.csv", retention_rows)

    print_table(
        "Statistics by exact observation count",
        exact_rows,
        columns=(
            "scene",
            "n_observations",
            "n_segments",
            "mean_points",
            "median_points",
            "min_points",
            "max_points",
        ),
    )

    print_table(
        "Statistics by observation bracket",
        bracket_rows,
        columns=(
            "scene",
            "observation_bracket",
            "n_segments",
            "mean_points",
            "median_points",
            "min_points",
            "max_points",
        ),
    )

    print_table(
        "Correlation between observations and point support",
        correlation_rows,
        columns=(
            "scene",
            "n_segments",
            "pearson_observations_points",
            "spearman_observations_points",
        ),
    )

    print_table(
        "K_min retention",
        retention_rows,
        columns=(
            "scene",
            "k_min",
            "retained_segments",
            "segment_retention_ratio",
            "retained_points",
            "point_retention_ratio",
        ),
    )

    if not args.no_plots:
        create_plots(
            all_stats,
            exact_rows,
            bracket_rows,
            retention_rows,
            output_dir,
        )

    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
