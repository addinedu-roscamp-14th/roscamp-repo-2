import numpy as np
import pytest

from central_control.top_view import TopViewTransformer


def save_calibration(tmp_path):
    camera_matrix = tmp_path / "cameraMatrix.npy"
    dist_coeffs = tmp_path / "distCoeffs.npy"
    np.save(camera_matrix, np.eye(3, dtype=np.float64))
    np.save(dist_coeffs, np.zeros(5, dtype=np.float64))
    return camera_matrix, dist_coeffs


def test_disabled_transform_keeps_mock_frame_unchanged():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    assert TopViewTransformer({}).transform(frame) is frame


def test_enabled_transform_uses_configured_bev_size(tmp_path):
    camera_matrix, dist_coeffs = save_calibration(tmp_path)
    config = {
        "bev_enabled": True,
        "bev_camera_matrix_path": str(camera_matrix),
        "bev_dist_coeffs_path": str(dist_coeffs),
        "bev_src_points": [[0, 0], [639, 0], [639, 479], [0, 479]],
        "bev_width": 600,
        "bev_height": 400,
    }
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = (0, 255, 0)

    transformed = TopViewTransformer(config).transform(frame)

    assert transformed.shape == (400, 600, 3)
    assert np.count_nonzero(transformed[:, :, 1]) > 0


def test_enabled_transform_requires_four_source_points(tmp_path):
    camera_matrix, dist_coeffs = save_calibration(tmp_path)
    with pytest.raises(ValueError, match="exactly four"):
        TopViewTransformer(
            {
                "bev_enabled": True,
                "bev_camera_matrix_path": str(camera_matrix),
                "bev_dist_coeffs_path": str(dist_coeffs),
                "bev_src_points": [[0, 0], [1, 0], [1, 1]],
            }
        )
