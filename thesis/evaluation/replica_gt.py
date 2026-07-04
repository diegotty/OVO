import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ai-made, human-proofed
# reads the gt instances of objects and saves them in a JSON format

IGNORED_CLASS_IDS = {-1, -2, 256}
@dataclass
class GTInstance:
    """
    One ground-truth object from Replica.

    Example:
        instance_id = 37
        class_name = "chair"
        sizes = [0.6, 0.9, 0.6]
    """
    instance_id: int
    class_id: int
    class_name: str
    ignored: bool

    # Bounding-box information provided by Replica.
    local_center: np.ndarray
    sizes: np.ndarray
    translation: np.ndarray
    rotation_xyzw: np.ndarray

    def to_dict(self) -> dict:
        """
        Convert NumPy arrays into lists so the object can be saved as JSON.
        """
        return {
            "instance_id": self.instance_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "ignore": self.ignored,
            "local_center": self.local_center.tolist(),
            "sizes": self.sizes.tolist(),
            "translation": self.translation.tolist(),
            "rotation_xyzw": self.rotation_xyzw.tolist(),
        }


def normalize_scene_name(scene_name: str) -> str:
    """
    Convert different spellings of the same scene into the same string.

    Examples:
        "office0"  -> "office0"
        "office_0" -> "office0"
        "Office-0" -> "office0"
    """
    return re.sub(pattern=r"[^a-z0-9]", repl="", string=scene_name.lower())


def find_info_semantic_file(replica_root: Path, scene_name: str) -> Path:
    """
    Find the official Replica `info_semantic.json` for one OVO scene.
    OVO calls a scene: office0
    Official Replica may call the same scene: office_0
    This function resolves that naming difference automatically.
    """
    replica_root = replica_root.expanduser().resolve()

    if not replica_root.exists():
        raise FileNotFoundError(
            f"Replica directory does not exist: {replica_root}"
        )

    requested_name = normalize_scene_name(scene_name)

    matches = []

    for info_file in replica_root.rglob("info_semantic.json"):
        # Expected path:
        #
        # replica_root/
        # └── office_0/
        #     └── habitat/
        #         └── info_semantic.json
        #
        # info_file.parent        -> habitat
        # info_file.parent.parent -> office_0
        official_scene_name = info_file.parent.parent.name
        if normalize_scene_name(official_scene_name) == requested_name:
            matches.append(info_file)

    if len(matches) == 0:
        available_scenes = sorted({
                info_file.parent.parent.name
                for info_file in replica_root.rglob("info_semantic.json")
        })
        raise FileNotFoundError(
            f"could not find annotations for scene '{scene_name}'.\n"
            f"replica root: {replica_root}\n"
            f"available scenes: {available_scenes}"
        )

    if len(matches) > 1:
        match_list = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            f"more than one annotation file matched '{scene_name}':\n"
            f"{match_list}"
        )

    return matches[0]


def read_vector(value, expected_length: int, field_name: str) -> np.ndarray:
    """
    Read a list from the JSON and validate that it has the expected size.
    Examples:
        a center must contain 3 numbers, a quaternion must contain 4 numbers
    """
    vector = np.asarray(value, dtype=np.float64)

    if vector.shape != (expected_length,):
        raise ValueError(
            f"{field_name} should contain {expected_length} numbers, "
            f"but its shape is {vector.shape}"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"{field_name} contains invalid values: {vector}"
        )

    return vector


def load_replica_instances(replica_root: Path, scene_name: str) -> list[GTInstance]:
    """
    Load every annotated object in one official Replica scene.

    replica_root: Root directory of the official Replica v1.0 dataset.
    scene_name: OVO scene name, for example "office0".

    returns a list of GTInstance objects.
    """
    info_file = find_info_semantic_file(replica_root=replica_root, scene_name=scene_name)

    with info_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "classes" not in data:
        raise KeyError(f"The file does not contain a 'classes' field: {info_file}")

    if "objects" not in data:
        raise KeyError(f"The file does not contain an 'objects' field: {info_file}")

    # Build:
    #
    # class ID -> class name
    #
    # Example:
    # 20 -> "chair"
    class_names = {
        int(class_info["id"]): str(class_info["name"])
        for class_info in data["classes"]
    }

    instances = []
    for object_info in data["objects"]:
        instance_id = int(object_info["id"])
        class_id = int(object_info["class_id"])

        if class_id in IGNORED_CLASS_IDS:
            class_name = 'unknown'
            ignored = True
        elif class_id not in class_names:
            raise KeyError(
                f"Object {instance_id} references unknown class ID "
                f"{class_id}"
            )
        else:
            class_name = class_names[class_id]
            ignored = False

        bbox = object_info["oriented_bbox"]
        local_box = bbox["abb"]
        orientation = bbox["orientation"]

        instance = GTInstance(
            instance_id=instance_id,
            class_id=class_id,
            class_name=class_name,
            ignored=ignored,
            local_center=read_vector(local_box["center"], expected_length=3, field_name=f"object {instance_id} center"),
            sizes=read_vector(local_box["sizes"], expected_length=3, field_name=f"object {instance_id} sizes"),
            translation=read_vector(orientation["translation"], expected_length=3, field_name=f"object {instance_id} translation"),
            rotation_xyzw=read_vector(orientation["rotation"], expected_length=4, field_name=f"object {instance_id} rotation"),
        )

        if np.any(instance.sizes < 0):
            raise ValueError(
                f"Object {instance_id} has negative box sizes: "
                f"{instance.sizes}"
            )

        instances.append(instance)

    if len(instances) == 0:
        raise ValueError(f"No objects were loaded from {info_file}")

    return instances


def save_replica_instances(instances: list[GTInstance], output_file: Path, scene_name: str) -> None:
    """
    save the cleaned annotations in a simpler JSON format
    avoids reading Replica's original JSON structure throughout the rest of the evaluation pipeline
    """
    output_file = output_file.expanduser().resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "scene_name": scene_name,
        "num_instances": len(instances),
        "instances": [
            instance.to_dict()
            for instance in instances
        ],
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

def load_prepared_instances(input_file: Path) -> list[GTInstance]:
    """
    load the simplified JSON produced by prepare_replica_gt.py
    """
    input_file = input_file.expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Prepared Replica GT file does not exist: {input_file}")

    with input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    instances = []
    for object_info in data["instances"]:
        instance = GTInstance(
            instance_id=int(object_info["instance_id"]),
            class_id=int(object_info["class_id"]),
            class_name=str(object_info["class_name"]),
            ignored=bool(object_info["ignore"]),
            local_center=np.asarray(object_info["local_center"], dtype=np.float64),
            sizes=np.asarray(object_info["sizes"], dtype=np.float64),
            translation=np.asarray(object_info["translation"], dtype=np.float64),
            rotation_xyzw=np.asarray(object_info["rotation_xyzw"], dtype=np.float64),
        )
        instances.append(instance)
    return instances
