import cv2
import numpy as np
import math

EMOTION_COLORS = {
    "happy":    (80, 220, 0),
    "sad":      (200, 100, 50),
    "angry":    (50, 50, 220),
    "fear":     (200, 50, 200),
    "surprise": (0, 220, 220),
    "disgust":  (0, 130, 220),
    "neutral":  (180, 180, 180),
}

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_MED   = 0.55
_FONT_SMALL = 0.38
_CORNER_LEN = 24
_THICKNESS  = 2


def _fill_rect(frame, x1, y1, x2, y2, color, alpha):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    bg = np.full(roi.shape, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0)


def _draw_corners(frame, x, y, w, h, color):
    arm = min(_CORNER_LEN, w // 2, h // 2)
    if arm <= 0:
        return
    thick = _THICKNESS + 1
    # Corner segments
    pts = [
        ((x, y),         (x + arm, y)),
        ((x, y),         (x, y + arm)),
        ((x + w, y),     (x + w - arm, y)),
        ((x + w, y),     (x + w, y + arm)),
        ((x, y + h),     (x + arm, y + h)),
        ((x, y + h),     (x, y + h - arm)),
        ((x + w, y + h), (x + w - arm, y + h)),
        ((x + w, y + h), (x + w, y + h - arm)),
    ]
    for p1, p2 in pts:
        cv2.line(frame, p1, p2, color, thick, cv2.LINE_AA)
    # Dot at each corner
    for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)


def _draw_confidence_arc(frame, cx, cy, radius, score, color):
    """Draw a circular arc representing confidence (0–1) around a center point."""
    # Dark background ring
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, 360,
                (50, 50, 50), 2, cv2.LINE_AA)
    # Colored arc for score
    end_angle = int(360 * score)
    if end_angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, end_angle,
                    color, 2, cv2.LINE_AA)
    # Percentage text centered
    pct = f"{int(score * 100)}"
    (tw, th), _ = cv2.getTextSize(pct, _FONT, _FONT_SMALL, 1)
    cv2.putText(frame, pct, (cx - tw // 2, cy + th // 2),
                _FONT, _FONT_SMALL, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "%", (cx + tw // 2 - 1, cy + th // 2 - 4),
                _FONT, _FONT_SMALL - 0.1, color, 1, cv2.LINE_AA)


def _draw_face_hud(frame, x, y, w, h, emotion, score, color):
    arc_r  = 18
    pad    = 6
    badge_h = 28

    # Badge sits below the face box
    badge_y1 = y + h + pad
    badge_y2 = badge_y1 + badge_h

    # Badge width: arc + text
    label = emotion.upper()
    (lw, lh), _ = cv2.getTextSize(label, _FONT, _FONT_MED, 1)
    badge_w = arc_r * 2 + 10 + lw + 14
    badge_x1 = x
    badge_x2 = badge_x1 + badge_w

    # Clamp to frame
    fh, fw = frame.shape[:2]
    badge_x2 = min(badge_x2, fw - 1)
    badge_y2 = min(badge_y2, fh - 1)

    # Badge background
    _fill_rect(frame, badge_x1, badge_y1, badge_x2, badge_y2, (10, 10, 10), 0.72)
    # Thin colored top border on badge
    cv2.line(frame, (badge_x1, badge_y1), (badge_x2, badge_y1), color, 1, cv2.LINE_AA)

    # Confidence arc inside badge
    arc_cx = badge_x1 + arc_r + 6
    arc_cy = badge_y1 + badge_h // 2
    _draw_confidence_arc(frame, arc_cx, arc_cy, arc_r - 4, score, color)

    # Emotion name
    tx = arc_cx + arc_r + 6
    ty = badge_y1 + badge_h // 2 + lh // 2
    cv2.putText(frame, label, (tx + 1, ty + 1), _FONT, _FONT_MED,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), _FONT, _FONT_MED,
                color, 1, cv2.LINE_AA)

    # Thin colored line connecting box bottom-left to badge
    cv2.line(frame, (x, y + h), (badge_x1, badge_y1), color, 1, cv2.LINE_AA)


def draw_overlay(frame, results, fps=0.0):
    fh, fw = frame.shape[:2]

    # ── Header ──────────────────────────────────────────────────────
    _fill_rect(frame, 0, 0, fw, 36, (5, 5, 5), 0.75)
    cv2.line(frame, (0, 36), (fw, 36), (80, 220, 0), 1)
    cv2.putText(frame, "EMOTION DETECTOR", (12, 25),
                _FONT, _FONT_MED, (80, 220, 0), 1, cv2.LINE_AA)
    fps_txt = f"FPS  {fps:.1f}"
    (fw2, _), _ = cv2.getTextSize(fps_txt, _FONT, _FONT_SMALL, 1), None
    cv2.putText(frame, fps_txt, (fw - 80, 25),
                _FONT, _FONT_SMALL, (150, 150, 150), 1, cv2.LINE_AA)

    # ── Face results ─────────────────────────────────────────────────
    for r in results:
        x, y, w, h = r["box"]
        emotion    = r["emotion"]
        score      = r["score"]
        color      = EMOTION_COLORS.get(emotion, (200, 200, 200))

        _draw_corners(frame, x, y, w, h, color)
        _draw_face_hud(frame, x, y, w, h, emotion, score, color)

    return frame
