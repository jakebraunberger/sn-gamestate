import json
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
CAMERA_SCALAR_KEYS = (
    "pan_degrees", "tilt_degrees", "roll_degrees", "x_focal_length", "y_focal_length",
)
CAMERA_COLUMNS = (
    "video_id", "frame", "image_id", "calibration_backend", "image_width", "image_height",
    "camera_parameters_valid", "camera_parameters_reused",
    *CAMERA_SCALAR_KEYS, "position_x_m", "position_y_m", "position_z_m",
    "principal_point_x", "principal_point_y", "parameters_json", "estimated_parameters_json",
)


def json_camera_value(value):
    """Keep the backend schema, converting NumPy values and nonfinite numbers for JSON."""
    if isinstance(value, np.ndarray):
        return json_camera_value(value.tolist())
    if isinstance(value, np.generic):
        return json_camera_value(value.item())
    if isinstance(value, dict):
        return {key: json_camera_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_camera_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def camera_values(parameters):
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("Camera parameters must be a dictionary or None.")
    parameters = json_camera_value(parameters or {})
    position = parameters.get("position_meters")
    principal = parameters.get("principal_point")
    if position is not None and (not isinstance(position, list) or len(position) != 3):
        raise ValueError("Camera position_meters must contain three coordinates.")
    if principal is not None and (not isinstance(principal, list) or len(principal) != 2):
        raise ValueError("Camera principal_point must contain two coordinates.")
    values = {key: parameters.get(key) for key in CAMERA_SCALAR_KEYS}
    values.update(zip(("position_x_m", "position_y_m", "position_z_m"), position or [None] * 3))
    values.update(zip(("principal_point_x", "principal_point_y"), principal or [None] * 2))
    valid = all(isinstance(value, (int, float)) and np.isfinite(value) for value in values.values())
    valid = valid and values["x_focal_length"] > 0 and values["y_focal_length"] > 0
    return values, bool(valid), json.dumps(parameters, allow_nan=False) if parameters else None


def homography_values(homography: Optional[np.ndarray]) -> list:
    if homography is None:
        return [np.nan] * 9
    return np.asarray(homography, dtype=float).reshape(3, 3).ravel().tolist()


class HomographyArchive:
    def __init__(self, export_dir: str):
        self.export_dir = Path(export_dir)
        self._initialized_videos = set()
        self._initialized_camera_videos = set()

    def reset(self):
        self._initialized_videos.clear()
        self._initialized_camera_videos.clear()

    def write(
        self,
        metadatas: pd.DataFrame,
        homographies: Sequence[Optional[np.ndarray]],
        estimated_homographies: Sequence[Optional[np.ndarray]],
        reused: Sequence[bool],
        *,
        camera_parameters: Optional[Sequence[Optional[dict]]] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        calibration_backend: Optional[str] = None,
    ):
        if not (
            len(metadatas)
            == len(homographies)
            == len(estimated_homographies)
            == len(reused)
        ):
            raise ValueError("Homography archive inputs must contain one value per image.")
        if camera_parameters is not None:
            if len(camera_parameters) != len(metadatas):
                raise ValueError("Camera archive inputs must contain one value per image.")
            if (image_width is None or image_height is None
                    or not np.isfinite([image_width, image_height]).all()
                    or min(image_width, image_height) <= 0 or not calibration_backend):
                raise ValueError("Camera archival requires positive image dimensions and a calibration backend.")
        if metadatas.empty:
            return
        missing_columns = {"video_id", "frame"} - set(metadatas.columns)
        if missing_columns:
            raise ValueError(
                f"Image metadata is missing required homography archive columns: "
                f"{sorted(missing_columns)}"
            )

        rows, camera_rows = [], []
        for index, ((image_id, metadata), homography, estimated, was_reused) in enumerate(zip(
            metadatas.iterrows(), homographies, estimated_homographies, reused
        )):
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
            if camera_parameters is not None:
                camera, valid, serialized = camera_values(camera_parameters[index])
                camera_rows.append({
                    "video_id": metadata["video_id"], "frame": metadata["frame"], "image_id": image_id,
                    "calibration_backend": calibration_backend,
                    "image_width": image_width, "image_height": image_height,
                    "camera_parameters_valid": valid, "camera_parameters_reused": bool(was_reused),
                    **camera, "parameters_json": serialized,
                    "estimated_parameters_json": None if was_reused else serialized,
                })

        self.export_dir.mkdir(parents=True, exist_ok=True)
        rows_by_video = pd.DataFrame(rows, columns=ARCHIVE_COLUMNS).groupby(
            "video_id", sort=False, dropna=False
        )
        cameras = pd.DataFrame(camera_rows, columns=CAMERA_COLUMNS)
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
            if camera_parameters is not None:
                camera_path = self.export_dir / f"camera_{filename}.csv"
                first_camera_write = video_id not in self._initialized_camera_videos
                cameras.loc[cameras["video_id"].eq(video_id)].to_csv(
                    camera_path, mode="w" if first_camera_write else "a",
                    header=first_camera_write, index=False,
                )
                self._initialized_camera_videos.add(video_id)
            self._initialized_videos.add(video_id)
