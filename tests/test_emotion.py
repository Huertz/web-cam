import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from emotion import EmotionDetector


def test_detect_returns_empty_list_when_no_faces():
    mock_fer = MagicMock()
    mock_fer.detect_emotions.return_value = []
    with patch("emotion.FER", return_value=mock_fer):
        detector = EmotionDetector(use_mtcnn=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result == []


def test_detect_returns_correct_structure():
    mock_fer = MagicMock()
    mock_fer.detect_emotions.return_value = [
        {
            "box": (10, 20, 100, 100),
            "emotions": {
                "happy": 0.9, "sad": 0.05, "angry": 0.02,
                "fear": 0.01, "surprise": 0.01, "disgust": 0.005, "neutral": 0.005,
            },
        }
    ]
    with patch("emotion.FER", return_value=mock_fer):
        detector = EmotionDetector(use_mtcnn=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert len(result) == 1
        assert result[0]["box"] == (10, 20, 100, 100)
        assert result[0]["emotion"] == "happy"
        assert result[0]["score"] == pytest.approx(0.9)


def test_detect_picks_dominant_emotion():
    mock_fer = MagicMock()
    mock_fer.detect_emotions.return_value = [
        {
            "box": (0, 0, 50, 50),
            "emotions": {
                "happy": 0.1, "sad": 0.7, "angry": 0.1,
                "fear": 0.02, "surprise": 0.03, "disgust": 0.02, "neutral": 0.03,
            },
        }
    ]
    with patch("emotion.FER", return_value=mock_fer):
        detector = EmotionDetector(use_mtcnn=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result[0]["emotion"] == "sad"
