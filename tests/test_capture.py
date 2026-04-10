import pytest
from unittest.mock import MagicMock, patch
import numpy as np


def test_camera_opens_successfully():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=mock_cap):
        from capture import CameraCapture
        cam = CameraCapture()
        assert cam is not None


def test_camera_raises_if_not_opened():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=mock_cap):
        from capture import CameraCapture
        with pytest.raises(RuntimeError, match="No se pudo abrir la cámara índice 0"):
            CameraCapture()


def test_read_returns_frame():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, fake_frame)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        from capture import CameraCapture
        cam = CameraCapture()
        ok, frame = cam.read()
        assert ok is True
        assert frame.shape == (480, 640, 3)


def test_release_calls_cap_release():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=mock_cap):
        from capture import CameraCapture
        cam = CameraCapture()
        cam.release()
        mock_cap.release.assert_called_once()
