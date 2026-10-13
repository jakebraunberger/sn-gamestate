import json

import numpy as np
import pandas as pd
import pytest

from sn_gamestate.calibration.homography_archive import HomographyArchive, camera_values
from sn_gamestate.dataset.image_folder import ImageFolderDataset


def test_archives_matrices_per_video_and_appends_frames(tmp_path):
    archive = HomographyArchive(str(tmp_path))
    homography = np.arange(9, dtype=float).reshape(3, 3)
    metadata = pd.DataFrame(
        {
            "video_id": [10, 10, 20],
            "frame": [1, 2, 1],
        },
        index=[101, 102, 201],
    )

    archive.write(
        metadata,
        [homography, None, homography],
        [homography, None, homography],
        [False, False, False],
    )
    archive.write(
        metadata.iloc[[1]],
        [homography],
        [None],
        [True],
    )

    video_10 = pd.read_csv(tmp_path / "homography_10.csv")
    video_20 = pd.read_csv(tmp_path / "homography_20.csv")
    assert video_10["frame"].tolist() == [1, 2, 2]
    assert video_10["image_id"].tolist() == [101, 102, 102]
    assert video_10.loc[0, "h00"] == 0
    assert video_10.loc[0, "h22"] == 8
    assert not video_10.loc[1, "homography_valid"]
    assert video_10.loc[2, "homography_reused"]
    assert pd.isna(video_10.loc[2, "estimated_h00"])
    assert video_20["frame"].tolist() == [1]


def test_reset_starts_a_fresh_archive_for_each_video(tmp_path):
    archive = HomographyArchive(str(tmp_path))
    metadata = pd.DataFrame({"video_id": [10], "frame": [1]}, index=[101])
    homography = np.eye(3)

    archive.write(metadata, [homography], [homography], [False])
    archive.reset()
    archive.write(metadata, [homography], [homography], [False])

    rows = pd.read_csv(tmp_path / "homography_10.csv")
    assert len(rows) == 1


def camera_parameters():
    return {
        "pan_degrees": np.float64(2.0), "tilt_degrees": 75.0, "roll_degrees": 0.0,
        "x_focal_length": 1600.0, "y_focal_length": 1600.0,
        "position_meters": np.array([0.0, 60.0, -10.0]),
        "principal_point": [960.0, 540.0], "rotation_matrix": np.eye(3),
        "radial_distortion": np.zeros(6),
    }


def test_camera_archive_preserves_parameters_missing_frames_and_reuse(tmp_path):
    archive = HomographyArchive(str(tmp_path))
    metadata = pd.DataFrame({"video_id": ["a", "a", "b"], "frame": [1, 2, 1]}, index=[1, 2, 3])
    camera = camera_parameters()
    options = dict(image_width=1920, image_height=1080, calibration_backend="nbjw_calib")
    archive.write(metadata, [np.eye(3), None, np.eye(3)], [np.eye(3), None, np.eye(3)],
                  [False] * 3, camera_parameters=[camera, {}, camera], **options)
    archive.write(metadata.iloc[[1]], [np.eye(3)], [None], [True],
                  camera_parameters=[camera], **options)

    rows = pd.read_csv(tmp_path / "camera_a.csv")
    assert rows["frame"].tolist() == [1, 2, 2]
    assert rows["image_id"].tolist() == [1, 2, 2]
    assert rows["camera_parameters_valid"].tolist() == [True, False, True]
    assert rows["camera_parameters_reused"].tolist() == [False, False, True]
    assert rows["position_z_m"].iloc[0] == -10.0
    assert rows["image_width"].tolist() == [1920] * 3
    assert rows["image_height"].tolist() == [1080] * 3
    assert rows["calibration_backend"].tolist() == ["nbjw_calib"] * 3
    saved = json.loads(rows["parameters_json"].iloc[0])
    np.testing.assert_array_equal(saved["rotation_matrix"], camera["rotation_matrix"])
    assert saved["position_meters"] == [0.0, 60.0, -10.0]
    assert saved["radial_distortion"] == [0.0] * 6
    assert rows["estimated_parameters_json"].iloc[0] == rows["parameters_json"].iloc[0]
    assert pd.isna(rows["parameters_json"].iloc[1])
    assert pd.isna(rows["estimated_parameters_json"].iloc[2])
    assert len(pd.read_csv(tmp_path / "camera_b.csv")) == 1
    homographies = pd.read_csv(tmp_path / "homography_a.csv")
    pd.testing.assert_frame_equal(rows[["frame", "image_id"]], homographies[["frame", "image_id"]])


def test_camera_archive_can_start_after_homography_only_write_and_reset(tmp_path):
    archive = HomographyArchive(str(tmp_path))
    metadata = pd.DataFrame({"video_id": [10], "frame": [1]}, index=[1])
    archive.write(metadata, [np.eye(3)], [np.eye(3)], [False])
    options = dict(camera_parameters=[camera_parameters()], image_width=1920,
                   image_height=1080, calibration_backend="baseline")
    archive.write(metadata, [np.eye(3)], [np.eye(3)], [False], **options)
    assert len(pd.read_csv(tmp_path / "camera_10.csv")) == 1
    archive.reset()
    archive.write(metadata, [np.eye(3)], [np.eye(3)], [False], **options)
    assert len(pd.read_csv(tmp_path / "camera_10.csv")) == 1
    assert len(pd.read_csv(tmp_path / "homography_10.csv")) == 1


def test_nonfinite_camera_is_flagged_and_json_uses_null():
    camera = camera_parameters()
    camera["position_meters"][2] = np.nan
    camera["x_focal_length"] = np.inf
    _, valid, serialized = camera_values(camera)
    assert not valid
    saved = json.loads(serialized)
    assert saved["position_meters"][2] is None
    assert saved["x_focal_length"] is None
    assert "NaN" not in serialized and "Infinity" not in serialized


@pytest.mark.parametrize("options, message", [
    (dict(camera_parameters=[]), "one value per image"),
    (dict(camera_parameters=[{}]), "positive image dimensions"),
    (dict(camera_parameters=[{}], image_width=0, image_height=1080,
          calibration_backend="nbjw_calib"), "positive image dimensions"),
])
def test_camera_archive_rejects_invalid_inputs(tmp_path, options, message):
    archive = HomographyArchive(str(tmp_path))
    metadata = pd.DataFrame({"video_id": [10], "frame": [1]})
    with pytest.raises(ValueError, match=message):
        archive.write(metadata, [np.eye(3)], [np.eye(3)], [False], **options)
    assert not list(tmp_path.glob("*.csv"))


def test_nbjw_process_archives_new_reused_and_missing_camera_parameters(tmp_path):
    from sn_gamestate.calibration.nbjw_calib import NBJW_Calib

    class Calibration:
        homography = None

        def update(self, predictions):
            pass

        def get_homography_from_ground_plane(self, **kwargs):
            return self.homography

        def heuristic_voting(self):
            return {"cam_params": camera_parameters()}

    module = NBJW_Calib(1920, 1080, 1, True, homography_export_dir=str(tmp_path))
    module.cam = Calibration()
    for frame in range(1, 4):
        module.cam.homography = np.eye(3) if frame == 2 else None
        metadata = pd.DataFrame({"video_id": ["test"], "frame": [frame], "keypoints": [{}]},
                                index=[frame])
        detections = pd.DataFrame({"bbox_ltwh": [np.array([1.0, 2.0, 3.0, 4.0])]})
        _, output = module.process(None, detections, metadata)
        assert output["homography_reused"].iloc[0] == (frame == 3)
    cameras = pd.read_csv(tmp_path / "camera_test.csv")
    assert cameras["camera_parameters_valid"].tolist() == [False, True, True]
    assert cameras["camera_parameters_reused"].tolist() == [False, False, True]
    assert pd.isna(cameras["parameters_json"].iloc[0])
    assert cameras["parameters_json"].iloc[1] == cameras["parameters_json"].iloc[2]
    assert pd.isna(cameras["estimated_parameters_json"].iloc[2])
    module.reset()
    assert module.last_h is None and module.last_params is None


def test_image_folder_dataset_builds_ordered_video_metadata(tmp_path):
    for filename in ("frame_10.jpg", "frame_2.jpg", "notes.json"):
        (tmp_path / filename).touch()

    dataset = ImageFolderDataset(str(tmp_path), str(tmp_path))
    tracking_set = dataset.sets["val"]

    assert tracking_set.video_metadatas.index.tolist() == [tmp_path.name]
    assert tracking_set.image_metadatas["name"].tolist() == ["frame_2", "frame_10"]
    assert tracking_set.image_metadatas["frame"].tolist() == [0, 1]
    assert tracking_set.image_metadatas["video_id"].tolist() == [tmp_path.name] * 2
    assert tracking_set.image_metadatas["file_path"].tolist() == [
        str((tmp_path / "frame_2.jpg").resolve()),
        str((tmp_path / "frame_10.jpg").resolve()),
    ]


def test_image_folder_dataset_rejects_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="No supported image files"):
        ImageFolderDataset(str(tmp_path), str(tmp_path))
