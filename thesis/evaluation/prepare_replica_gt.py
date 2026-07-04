import argparse
import os
import sys
from pathlib import Path
from replica_gt import load_replica_instances, save_replica_instances

repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))

# Replica scenes used by OVO
DEFAULT_SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]

def parse_arguments() -> argparse.Namespace:
    """
    Read command-line arguments.
    Example:
        python scripts/prepare_replica_gt.py \
            --replica-root ~/datasets/replica_v1 \
            --scenes office0 room0
    """
    parser = argparse.ArgumentParser(
        description=("Load Replica v1.0 instance annotations and save them in a simplified JSON format.")
    )

    parser.add_argument("--replica-root", type=Path, default=None,
        help=(
            "Path to the official Replica v1.0 dataset. "
            "If omitted, REPLICA_V1_ROOT is used."
        ),
    )

    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES,
        help=(
            "Scenes to prepare. "
            "Default: all Replica scenes used by OVO."
        ),
    )

    parser.add_argument( "--output-dir", type=Path, default=Path("data/evaluation/replica"),
        help=(
            "Directory where the simplified GT files will be saved."
        ),
    )

    return parser.parse_args()


def get_replica_root(command_line_path: Path | None) -> Path:
    """
    Determine where the official Replica dataset is stored.
    Priority:
    1. --replica-root command-line argument
    2. REPLICA_V1_ROOT environment variable
    """
    if command_line_path is not None:
        replica_root = command_line_path
    else:
        environment_path = os.environ.get("REPLICA_V1_ROOT")

        if environment_path is None:
            raise RuntimeError(
                "Replica root was not provided.\n"
                "Either pass:\n"
                "  --replica-root /path/to/replica_v1\n"
                "or set:\n"
                "  export REPLICA_V1_ROOT=/path/to/replica_v1"
            )
        replica_root = Path(environment_path)
    replica_root = replica_root.expanduser().resolve()

    if not replica_root.exists():
        raise FileNotFoundError(f"Replica root does not exist: {replica_root}")
    return replica_root


def prepare_scene(replica_root: Path, scene_name: str, output_directory: Path) -> None:
    """
    Load and save the GT instances for one scene.

    Example:
        office0
            ↓
        official Replica office_0/info_semantic.json
            ↓
        data/evaluation/replica/office0_instances.json
    """
    instances = load_replica_instances(replica_root=replica_root, scene_name=scene_name)

    output_file = (output_directory/ f"{scene_name}_instances.json")

    save_replica_instances(instances=instances, output_file=output_file, scene_name=scene_name)

    print(f"[{scene_name}]")
    print(f"  loaded instances: {len(instances)}")
    print(f"  saved to:         {output_file}")
    print()


def main() -> None:
    """
    Prepare all requested Replica scenes.
    """
    arguments = parse_arguments()
    replica_root = get_replica_root(arguments.replica_root)

    output_directory = (arguments.output_dir.expanduser().resolve())

    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Replica root:    {replica_root}")
    print(f"Output directory: {output_directory}")
    print(f"Scenes:          {arguments.scenes}")
    print()

    for scene_name in arguments.scenes:
        try:
            prepare_scene(replica_root=replica_root, scene_name=scene_name, output_directory=output_directory)

        except Exception as error:
            print(f"[{scene_name}] FAILED")
            print(f"  {type(error).__name__}: {error}")
            print()
            raise


if __name__ == "__main__":
    main()
