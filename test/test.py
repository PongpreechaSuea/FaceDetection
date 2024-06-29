import os
from src.detect import YOLOFaceImageDetector

IMAGE_PATH = "./pngtree-multiethnic-group-of-people-standing-in-front-of-background-image_2971665.jpg"

detector = YOLOFaceImageDetector()

with open(IMAGE_PATH, "rb") as image_file:
    image_data = image_file.read()

filename = os.path.basename(IMAGE_PATH)
result = detector.run(image_data, filename)
