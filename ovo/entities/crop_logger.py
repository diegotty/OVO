# TODO add docs ?
import numpy as np
from pathlib import Path
from torchvision.utils import save_image
from PIL import Image
class CropLogger:
    """
    Asynchronously handles a crop cache consistent with OVO's top-k views for each tracked 3D segment.
    """
    def __init__(self, cache_dir : Path, queue_size : int = 4):
        self.cache_dir = cache_dir

        # in case we want to save full sequence frames aswell
        self.frames_dir = cache_dir / "frames"
        self.segments_dir = cache_dir / "segments"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)

    def add_keyframe(self, kf_id, segment_ids, crops):
        # TODO check are crops already aligned ?
        for segment_id, segment_crops in (zip(segment_ids, crops):
            view_path = self.segments_dir / f"segment_{segment_id}" / f"kf_{kf_id}"
            view_path.mkdir(parents=True, exist_ok=True)
            save_image(segment_crops[:3].clamp(0, 1), view_path / "masked.png")
            save_image(segment_crops[3:].clamp(0, 1), view_path / "bbox.png")

    def remove_keyframe(self, segment_id, kf_id):
        view_path = {self.segments_dir / f"segment_{segment_id}" / f"kf_{kf_id}"
        if view_path.exists():
            shutil.rmtree(view_path)

        # no empty folders  !
        segment_path = view_path.parent
        if segment_path.exists() and not any(segment_path.iterdir()):
            segment_path.rmdir()
