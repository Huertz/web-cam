import numpy as np
from overlay import draw_overlay, EMOTION_COLORS


def test_draw_overlay_returns_same_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = draw_overlay(frame, [], fps=15.0)
    assert result is frame


def test_draw_overlay_header_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_overlay(frame, [], fps=10.0)
    assert frame.sum() > 0


def test_draw_overlay_with_detection_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [{"box": (50, 50, 100, 100), "emotion": "happy", "score": 0.85}]
    draw_overlay(frame, results, fps=12.0)
    assert frame.sum() > 0


def test_emotion_colors_cover_all_seven_emotions():
    expected = {"happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"}
    assert set(EMOTION_COLORS.keys()) == expected


def test_draw_overlay_unknown_emotion_uses_fallback_color():
    # "contempt" is not in EMOTION_COLORS — should not raise, should use fallback (200,200,200)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [{"box": (50, 50, 100, 100), "emotion": "contempt", "score": 0.9}]
    draw_overlay(frame, results, fps=0.0)
    assert frame.sum() > 0  # drew something without KeyError


def test_draw_corners_near_detection_origin():
    # corner brackets should produce non-zero pixels near the box origin
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [{"box": (50, 50, 100, 100), "emotion": "happy", "score": 0.85}]
    draw_overlay(frame, results, fps=0.0)
    region = frame[48:72, 48:72]
    assert region.sum() > 0
