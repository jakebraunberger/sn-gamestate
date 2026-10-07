import re
from pathlib import Path

import pandas as pd
from tracklab.datastruct import TrackingDataset, TrackingSet


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _natural_sort_key(path: Path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


class ImageFolderDataset(TrackingDataset):
    """Use the images in one folder as a video sequence for inference."""

    def __init__(self, dataset_path: str, video_path: str, *args, **kwargs):
        image_dir = Path(video_path)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Image folder does not exist: {image_dir}")

        image_paths = sorted(
            (
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=_natural_sort_key,
        )
        if not image_paths:
            raise ValueError(f"No supported image files found in {image_dir}")

        video_id = image_dir.name
        image_metadata = pd.DataFrame(
            [
                {
                    "id": frame,
                    "name": image_path.stem,
                    "video_id": video_id,
                    "frame": frame,
                    "nframes": len(image_paths),
                    "file_path": str(image_path.resolve()),
                }
                for frame, image_path in enumerate(image_paths)
            ]
        ).set_index("id", drop=False)
        video_metadata = pd.DataFrame(
            [{"id": video_id, "name": video_id, "nframes": len(image_paths)}]
        ).set_index("id", drop=False)

        tracking_set = TrackingSet(
            video_metadata,
            image_metadata,
            None,
            image_metadata.copy(),
        )
        super().__init__(dataset_path, {"val": tracking_set}, *args, **kwargs)
