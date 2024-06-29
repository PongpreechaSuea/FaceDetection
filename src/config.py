MODEL = "./model/yolov8n-face.onnx"
IMAGE_PATH = "image_debug.jpg"

IOU = 0.58
REG_MAX = 16
STRIDES = (8, 16, 32)

CONF = 0.3
IMGSIZE = 480
FREE_SPACE = 20 


HOST = "0.0.0.0"
PORT = 3000
BASE_URL = f"http://{HOST}:{PORT}"
SWAGGER = f"{BASE_URL}/docs"