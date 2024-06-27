import os
import json
import base64
import io
import cv2
import shutil
from PIL import Image
import numpy as np
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Encode
    
def _encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

# Get
    
def _get_file_name(photo_path: str = None):
    return os.path.basename(photo_path).rsplit(".", 1)

def _get_abs_path(path: str = None):
    return os.path.abspath(path)

def _get_bounding_box(box: dict, width: int, height: int):
    w = int(width * box['Width'])
    h = int(height * box['Height'])
    top = int(height * box['Top'])
    left = int(width * box['Left'])
    return left, top, h, w

# Move and Create

def _create_dir(path: str = None):
    os.makedirs(path, exist_ok=True)

def _move_file(source_path: str = None, destination_path: str = None):
    shutil.move(source_path, destination_path)

# Read 
    
def _read_image_pil(photo_path: str = ".jpg/.png"):
    image = Image.open(photo_path)
    stream = io.BytesIO()

    if 'exif' in image.info:
        exif = image.info['exif']
        image.save(stream, format=image.format, exif=exif)
    else:
        image.save(stream, format="JPEG")

    return image, stream.getvalue()

def _read_image_cv2(photo_path: str = ".jpg/.png"):
    return cv2.imread(photo_path)

# save

def _save_image_pil(image, file_path):
    image.save(file_path)

def _save_image_cv2(image, file_path):
    cv2.imwrite(file_path, image)

def _save_json(json_data: dict = None):
    json_path = json_data["json_output"]
    with open(json_path, "w") as outfile:
        json.dump(json_data, outfile, cls=NumpyArrayEncoder)