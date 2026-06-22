import torch
import argparse
import numpy as np
from pathlib import Path

def extract(checkpoint, output_dir):
    points3d = checkpoint["map_params"]
    points_ids = points3d["ids"].squeeze()
    ovo_params = checkpoint["ovo_map_params"]

    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)
    descriptors = []
    
    # extratcted segments' ids. the order will match with descriptors.npy
    extracted_ids = []
    segments_metadata = []
    segment_ids = to_numpy(ovo_params["ins_3d_ids"])

    for row, id in enumerate(segment_ids):
        segment_id = int(id)

        # the 3d points in the pointcloud that correspond to the current segment
        # pytorch moment: the comparison creates a boolean mask, and uses it as such for points3d
        segment_points = points3d[points_ids == segment_id]
        # segment is empty ! we dont track it
        if len(segment_points) == 0:
            continue

        descriptor = to_numpy(ovo_params[f"ins3d_{segment_id}_clip_feature"])
        descriptors.append(descriptor.astype(np.float32))
        extracted_ids.append(segment_id)

        keyframe_ids = to_numpy(ovo_params[f"ins3d_{segment_id}_keyframes_ids"])
        top_keyframes = to_numpy(ovo_params[f"ins3d_{segment_id}_top_kfs"])

        segment_metadata = {
            "id" : segment_id,
            "descriptor_row" : row,
            "points_file" : (f"segments/{segment_id}/points.npy"),
            "keyframe_ids" : keyframe_ids.tolist(),
            "top_views" : [],
        }

        # by setting k_top_views == 0 in the config.yaml, we can choose to use every observation
        # of the 3d segments. this mode leaves the heap empty, resulting in 0 best views (as its 
        # a information we get from the heap
        # also, top_kfs is CURRENTLY debug information (however, it is important for our methodology
        # so TODO: make it mandatory)
        if top_keyframes.size > 0:
            # -- WIP --
            top_keyframes = top_keyframes.reshape(-1, 2)
            # top_keyframes = sorted(top_keyframes.tolist(), key=lambda entry: entry[0], reverse = True)

        top_view_metadata = {
            "keyframe_id" : keyframe,
            "mask_area" : mask_area,
            "crop" : ,
            "mask" : ,
        }
        # dubious
        segment_metadata["top_views"].append(top_view_metadata)
        # -- WIP --

        segments_metadata.append(segment_metadata)
        segment_dir = segments_dir / str(segment_id)
        segment_dir.mkdir(exist_ok=True)
        (segment_dir / "crops").mkdir(exist_ok=True)
        np.save(segment_dir / "points.npy", segment_points.astype(np.float32))


    # segment order in both matches
    np.save(output_dir / "descriptors.npy", np.stack(descriptors).astype(np.float32))
    np.save(output_dir / "segment_ids.npy", np.stack(extracted_ids).astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    # requires a path as an argument when executing the file
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        print("give me a checkpoint file ....")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        print("wrong checkpoint format")
    extract(checkpoint, args.output_dir)
    return

if __name__ == "__main__":
   main()


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
