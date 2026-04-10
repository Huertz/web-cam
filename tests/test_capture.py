import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from capture import CameraCapture


def test_camera_opens_successfully():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch("capture.cv2.VideoCapture", return_value=mock_cap):
        cam = CameraCapture()
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def test_camera_raises_if_not_opened():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    with patch("capture.cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="No se pudo abrir la cámara índice 0"):
            CameraCapture()


def test_read_returns_frame():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, fake_frame)
    with patch("capture.cv2.VideoCapture", return_value=mock_cap):
        cam = CameraCapture()
        ok, frame = cam.read()
        assert ok is True
        assert frame.shape == (480, 640, 3)


def test_release_calls_cap_release():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch("capture.cv2.VideoCapture", return_value=mock_cap):
        cam = CameraCapture()
        cam.release()
        mock_cap.release.assert_called_once()
