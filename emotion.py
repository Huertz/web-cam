# fer>=25.x does not re-export FER from __init__.py; import from the submodule directly.
# Public API `from fer import FER` raises ImportError on this version.
from fer.fer import FER


class EmotionDetector:
    def __init__(self, use_mtcnn=True):
        print("Cargando modelo...")
        self._detector = FER(mtcnn=use_mtcnn)

    def detect(self, frame):
        raw = self._detector.detect_emotions(frame)
        results = []
        for r in raw:
            emotions = r["emotions"]
            if not emotions:
                continue
            dominant = max(emotions, key=emotions.get)
            results.append({
                "box": r["box"],
                "emotion": dominant,
                "score": emotions[dominant],
                "emotions": emotions,
            })
        return results
