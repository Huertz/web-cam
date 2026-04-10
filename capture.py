import cv2


class CameraCapture:
    def __init__(self, device_index=0, width=640, height=480):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara índice {device_index}. "
                "Verifica permisos en Ajustes del sistema → Privacidad y seguridad → Cámara."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()
