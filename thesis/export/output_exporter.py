import torch
import argparse
import shutil
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def extract(input_dir):
    scene_split = str(input_dir).rsplit('/', 1)
    scene = scene_split[-1]
    output_dir = REPO_ROOT / "exported" / scene
    checkpoint = input_dir / "ovo_map.ckpt"
    if not checkpoint.is_file():
        raise RuntimeError(f"give me a checkpoint file ....{checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("wrong checkpoint format")
    map_params = checkpoint["map_params"]
    xyz = to_numpy(map_params["xyz"])
    obj_ids = to_numpy(map_params["obj_ids"].reshape(-1))

    ovo_params = checkpoint["ovo_map_params"]
    source_frame_ids = to_numpy(ovo_params["frame_id"]).reshape(-1)

    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True, parents=True)
    descriptors = []
    
    # extratcted segments' ids. the order will match with descriptors.npy
    extracted_ids = []
    segments_metadata = []

    segment_ids = to_numpy(ovo_params["ins_3d_ids"])
    for _, id in enumerate(segment_ids):
        segment_id = int(id)
        top_views = []

        # the 3d points in the pointcloud that correspond to the current segment
        # pytorch moment: the comparison creates a boolean mask, and uses it as such for points3d
        segment_points = xyz[obj_ids == segment_id]
        # segment is empty ! we dont track it
        if len(segment_points) == 0:
            continue

        segment_dir = segments_dir / f"{segment_id:05d}"
        crop_dir = segment_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)

        descriptor_row = len(descriptors)
        descriptor = to_numpy(ovo_params[f"ins3d_{segment_id}_clip_feature"]).reshape(-1).astype(np.float32)
        descriptors.append(descriptor)
        extracted_ids.append(segment_id)

        kf_ids = to_numpy(ovo_params[f"ins3d_{segment_id}_keyframes_ids"]).reshape(-1).astype(np.int64)
        top_kfs = to_numpy(ovo_params[f"ins3d_{segment_id}_top_kfs"])
        top_kfs = sorted(top_kfs, key = lambda item: item[0], reverse=True)

        # by setting k_top_views == 0 in the config.yaml, we can choose to use every observation
        # of the 3d segments. this mode leaves the heap empty, resulting in 0 best views (as its 
        # a information we get from the heap

        for mask_area, kf_id in top_kfs:
            mask_area = int(mask_area)
            kf_id = int(kf_id)
           # copy crops from crop_cache 
            source_dir = input_dir / "crop_cache" / f"segment_{segment_id:05d}" / f"kf_{kf_id:05d}"
            relative_dest_dir = Path("segments") / f"{segment_id:05d}" / "crops" / f"kf_{kf_id:05d}"
            dest_dir = output_dir / relative_dest_dir
            dest_dir.mkdir(parents=True, exist_ok=True)

            filenames = ["masked.png", "bbox.png", "descriptor.npy"]
            for filename in filenames:
                source_file = source_dir / filename
                if not source_file.exists():
                    raise FileNotFoundError(f"missing cached crop file: {source_file}")
                else:
                    shutil.copy2( source_file, dest_dir / filename)

            raw_source_frame_id = source_frame_ids[kf_id]
            source_frame_deleted = (str(raw_source_frame_id) == "Deleted")
            source_frame_id = None if source_frame_deleted else int(raw_source_frame_id)
            top_views.append({
                "keyframe_id" : kf_id,
                "source_frame_id" : source_frame_id,
                "mask_area" : mask_area,
                "masked_crop_file" : (relative_dest_dir / "masked.png").as_posix(),
                "bbox_crop_file" : (relative_dest_dir / "bbox.png").as_posix(),
                "descriptor" : (relative_dest_dir / "descriptor.npy").as_posix()
            })

        segment_metadata = {
            "id" : segment_id,
            "descriptor_row" : descriptor_row,
            "points_file" : (Path("segments") / f"{segment_id:05d}" / "points.npy").as_posix(),
            "keyframe_ids" : kf_ids.tolist(),
            "top_views" : top_views,
        }

        segments_metadata.append(segment_metadata)
        np.save(segment_dir / "points.npy", segment_points.astype(np.float32))


    # segment order in both matches
    if not descriptors:
        raise RuntimeError("no descriptors ...")
    np.save(output_dir / "descriptors.npy", np.stack(descriptors).astype(np.float32))
    np.save(output_dir / "segment_ids.npy", np.stack(extracted_ids).astype(np.int64))

    scene_metadata = {
        "format_version" : 1,
        "segment_ids_file" : "segment_ids.npy",
        "descriptors_file" : "descriptors.npy",
        "segments" : segments_metadata
    }
    with open(output_dir / "scene.json", "w", encoding="utf-8") as file:
        json.dump(scene_metadata, file, indent=2)
    return True

def main():
    parser = argparse.ArgumentParser()
    # requires a path as an argument when executing the file
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.is_file():
        raise RuntimeError("give me a checkpoint file ....")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("wrong checkpoint format")

    input_dir = args.checkpoint.parent
    extract(input_dir=input_dir, output_dir=args.output_dir)
    return

if __name__ == "__main__":
   main()

 
