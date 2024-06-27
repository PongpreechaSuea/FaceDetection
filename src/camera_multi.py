import cv2
from src.base_camera import BaseCamera

class Camera(BaseCamera):
    def __init__(self):
        super().__init__()

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            _, img = camera.read()
            yield cv2.imencode('.jpg', img)[1].tobytes() , img


# if __name__ in "__main__":
#     camera = Camera()
#     camera.get_frame()