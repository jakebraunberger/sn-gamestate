import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


HOMOGRAPHY_KEYS = tuple(f"h{row}{column}" for row in range(3) for column in range(3))
ESTIMATED_HOMOGRAPHY_KEYS = tuple(f"estimated_{key}" for key in HOMOGRAPHY_KEYS)
ARCHIVE_COLUMNS = (
    "video_id",
    "frame",
    "image_id",
    "homography_valid",
    "homography_reused",
    *HOMOGRAPHY_KEYS,
    *ESTIMATED_HOMOGRAPHY_KEYS,
)


def homography_values(homography: Optional[np.ndarray]) -> list:
    if homography is None:
        return [np.nan] * 9
    return np.asarray(homography, dtype=float).reshape(3, 3).ravel().tolist()


class HomographyArchive:
    def __init__(self, export_dir: str):
        self.export_dir = Path(export_dir)
        self._initialized_videos = set()

    def reset(self):
        self._initialized_videos.clear()

    def write(
        self,
        metadatas: pd.DataFrame,
        homographies: Sequence[Optional[np.ndarray]],
        estimated_homographies: Sequence[Optional[np.ndarray]],
        reused: Sequence[bool],
    ):
        if not (
            len(metadatas)
            == len(homographies)
            == len(estimated_homographies)
            == len(reused)
        ):
            raise ValueError("Homography archive inputs must contain one value per image.")
        if metadatas.empty:
            return
        missing_columns = {"video_id", "frame"} - set(metadatas.columns)
        if missing_columns:
            raise ValueError(
                f"Image metadata is missing required homography archive columns: "
                f"{sorted(missing_columns)}"
            )

        rows = []
        for (image_id, metadata), homography, estimated, was_reused in zip(
            metadatas.iterrows(), homographies, estimated_homographies, reused
        ):
            values = homography_values(homography)
            estimated_values = homography_values(estimated)
            rows.append(
                {
                    "video_id": metadata["video_id"],
                    "frame": metadata["frame"],
                    "image_id": image_id,
                    "homography_valid": bool(np.isfinite(values).all()),
                    "homography_reused": bool(was_reused),
                    **dict(zip(HOMOGRAPHY_KEYS, values)),
                    **dict(zip(ESTIMATED_HOMOGRAPHY_KEYS, estimated_values)),
                }
            )

        self.export_dir.mkdir(parents=True, exist_ok=True)
        rows_by_video = pd.DataFrame(rows, columns=ARCHIVE_COLUMNS).groupby(
            "video_id", sort=False, dropna=False
        )
        for video_id, video_rows in rows_by_video:
            if pd.isna(video_id):
                raise ValueError("Image metadata contains a missing video_id.")
            filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video_id)).strip("._")
            if not filename:
                raise ValueError(f"Cannot make a homography archive filename for {video_id!r}.")
            path = self.export_dir / f"homography_{filename}.csv"
            is_first_write = video_id not in self._initialized_videos
            video_rows.to_csv(
                path,
                mode="w" if is_first_write else "a",
                header=is_first_write,
                index=False,
            )
            self._initialized_videos.add(video_id)
