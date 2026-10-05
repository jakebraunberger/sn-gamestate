import numpy as np
import pandas as pd

from sn_gamestate.calibration.homography_archive import HomographyArchive


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
