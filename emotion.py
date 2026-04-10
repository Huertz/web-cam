# fer>=25.x does not re-export FER from __init__.py; import from the submodule directly.
# Public API `from fer import FER` raises ImportError on this version.
from collections import deque, Counter
from fer.fer import FER


class EmotionSmoother:
    """Averages emotion scores over the last N frames to eliminate flickering."""

    def __init__(self, window=8):
        self._window = window
        # face_id (int) → deque of emotion dicts
        self._history: dict[int, deque] = {}

    def smooth(self, results):
        smoothed = []
        seen_ids = set()

        for i, r in enumerate(results):
            seen_ids.add(i)
            hist = self._history.setdefault(i, deque(maxlen=self._window))
            hist.append(r["emotions"])

            # Average each emotion score across the history window
            avg: dict[str, float] = {}
            for emo_dict in hist:
                for emo, score in emo_dict.items():
                    avg[emo] = avg.get(emo, 0.0) + score
            n = len(hist)
            avg = {k: v / n for k, v in avg.items()}

            dominant = max(avg, key=avg.get)
            smoothed.append({
                "box":     r["box"],
                "emotion": dominant,
                "score":   avg[dominant],
                "emotions": avg,
            })

        # Drop history for faces that disappeared
        gone = set(self._history) - seen_ids
        for k in gone:
            del self._history[k]

        return smoothed


class EmotionDetector:
    def __init__(self, use_mtcnn=True):
        print("Cargando modelo...")
        self._detector = FER(mtcnn=use_mtcnn)
        self._smoother = EmotionSmoother(window=8)

    def detect(self, frame):
        raw = self._detector.detect_emotions(frame)
        results = []
        for r in raw:
            emotions = r["emotions"]
            if not emotions:
                continue
            dominant = max(emotions, key=emotions.get)
            results.append({
                "box":      r["box"],
                "emotion":  dominant,
                "score":    emotions[dominant],
                "emotions": emotions,
            })
        return self._smoother.smooth(results)
