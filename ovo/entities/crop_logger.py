import shutil
from pathlib import Path
from torchvision.utils import save_image
class CropLogger:
    """
    Asynchronously handles a crop cache consistent with OVO's top-k views for each tracked 3D segment.
    """
    def __init__(self, cache_dir : Path, queue_size : int = 4):
        # [stuff]/crop_cache
        self.cache_dir = cache_dir

        # in case we want to save full sequence frames aswell
        # self.frames_dir = cache_dir / "frames"
        # self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print("woop woop! crop logger created")

    def add_keyframe(self, kf_id, segment_ids, crops):
        for segment_id, segment_crops in zip(segment_ids, crops):
            view_path = self.cache_dir / f"segment_{segment_id:05d}" / f"kf_{kf_id:05d}"
            view_path.mkdir(parents=True, exist_ok=True)
            save_image(segment_crops[:3].clamp(0, 1), view_path / "masked.png")
            save_image(segment_crops[3:].clamp(0, 1), view_path / "bbox.png")

    def remove_keyframe(self, segment_id, kf_id):
        view_path = self.cache_dir / f"segment_{segment_id:05d}" / f"kf_{kf_id:05d}"
        if view_path.exists():
            shutil.rmtree(view_path)

        # no empty folders  !
        segment_path = view_path.parent
        if segment_path.exists() and not any(segment_path.iterdir()):
            segment_path.rmdir()
