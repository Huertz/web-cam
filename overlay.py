import cv2
import numpy as np

EMOTION_COLORS = {
    "happy":    (80, 220, 0),
    "sad":      (200, 100, 50),
    "angry":    (50, 50, 220),
    "fear":     (200, 50, 200),
    "surprise": (0, 220, 220),
    "disgust":  (0, 130, 220),
    "neutral":  (180, 180, 180),
}

# Order for the emotion bars panel
EMOTION_ORDER = ["happy", "surprise", "neutral", "sad", "fear", "angry", "disgust"]

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = 0.42
_FONT_MED   = 0.55
_CORNER_LEN = 20
_THICKNESS  = 2


def _fill_rect(frame, x1, y1, x2, y2, color, alpha):
    """Draw a filled semi-transparent rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_corners(frame, x, y, w, h, color):
    arm = min(_CORNER_LEN, w // 2, h // 2)
    if arm <= 0:
        return
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
        cv2.line(frame, p1, p2, color, _THICKNESS)


def _draw_emotion_bars(frame, emotions, x, y, w):
    """Draw a panel of all emotion scores below the face box."""
    bar_h      = 12
    bar_gap    = 6
    label_w    = 62
    bar_max_w  = max(w, 140)
    panel_w    = label_w + bar_max_w + 8
    panel_h    = len(EMOTION_ORDER) * (bar_h + bar_gap) + 6
    px, py     = x, y + 6

    # Panel background
    _fill_rect(frame, px, py, px + panel_w, py + panel_h, (10, 10, 10), 0.65)

    for i, emo in enumerate(EMOTION_ORDER):
        score  = emotions.get(emo, 0.0)
        color  = EMOTION_COLORS.get(emo, (200, 200, 200))
        row_y  = py + 4 + i * (bar_h + bar_gap)

        # Label
        cv2.putText(frame, emo[:7].upper(), (px + 3, row_y + bar_h - 2),
                    _FONT, _FONT_SMALL - 0.05, (200, 200, 200), 1, cv2.LINE_AA)

        # Bar background
        bx = px + label_w
        cv2.rectangle(frame, (bx, row_y), (bx + bar_max_w, row_y + bar_h),
                      (50, 50, 50), -1)

        # Bar fill
        fill = int(bar_max_w * score)
        if fill > 0:
            cv2.rectangle(frame, (bx, row_y), (bx + fill, row_y + bar_h),
                          color, -1)

        # Percentage text
        pct = f"{int(score * 100)}%"
        cv2.putText(frame, pct, (bx + fill + 3, row_y + bar_h - 2),
                    _FONT, _FONT_SMALL - 0.05, (220, 220, 220), 1, cv2.LINE_AA)


def draw_overlay(frame, results, fps=0.0):
    h_frame, w_frame = frame.shape[:2]

    # ── Header bar ──────────────────────────────────────────────────
    _fill_rect(frame, 0, 0, w_frame, 38, (10, 10, 10), 0.70)
    header = f"EMOTION DETECTOR"
    fps_txt = f"FPS: {fps:.1f}"
    cv2.putText(frame, header, (12, 26), _FONT, _FONT_MED, (80, 220, 0), 1, cv2.LINE_AA)
    tw, _ = cv2.getTextSize(fps_txt, _FONT, _FONT_MED, 1)[0], None
    cv2.putText(frame, fps_txt, (w_frame - 90, 26), _FONT, _FONT_MED,
                (180, 180, 180), 1, cv2.LINE_AA)

    # ── Per-face results ─────────────────────────────────────────────
    for r in results:
        x, y, w, h    = r["box"]
        emotion        = r["emotion"]
        score          = r["score"]
        all_emotions   = r.get("emotions", {})
        color          = EMOTION_COLORS.get(emotion, (200, 200, 200))

        # Corner brackets
        _draw_corners(frame, x, y, w, h, color)

        # Dominant emotion label with background
        label   = f"{emotion.upper()}  {int(score * 100)}%"
        (lw, lh), _ = cv2.getTextSize(label, _FONT, _FONT_MED, 1)
        lx, ly  = x, max(y - 10, 40)
        _fill_rect(frame, lx - 2, ly - lh - 4, lx + lw + 6, ly + 4,
                   (10, 10, 10), 0.65)
        cv2.putText(frame, label, (lx + 1, ly + 1), _FONT, _FONT_MED,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, ly), _FONT, _FONT_MED,
                    color, 1, cv2.LINE_AA)

        # Emotion bars below the face box
        if all_emotions:
            _draw_emotion_bars(frame, all_emotions, x, y + h, w)

    return frame
