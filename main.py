import sys
import time

import cv2

from capture import CameraCapture
from emotion import EmotionDetector
from overlay import draw_overlay


def main():
    try:
        camera = CameraCapture()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    detector = EmotionDetector()
    prev_time = time.monotonic()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Error: no se pudo leer el frame.", file=sys.stderr)
                break

            results = detector.detect(frame)

            now = time.monotonic()
            delta = now - prev_time
            fps = 1.0 / delta if delta > 0 else 0.0
            prev_time = now

            draw_overlay(frame, results, fps)
            cv2.imshow("Emotion Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
