import numpy as np
import pandas as pd
import pytest

from sn_gamestate.calibration.homography_archive import HomographyArchive
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
