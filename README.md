# Face Detection API

This project is a FastAPI-based web service for face detection using the YOLOv8 model.

## Features
- Face detection from uploaded images
- Real-time face detection from camera feed
- Adjustable confidence threshold and input size via API
- Display detection results on images and video stream

## Requirements
- Python 3.x
- FastAPI
- OpenCV
- ONNX Runtime
- Pillow

## Installation
1. Clone repository:
    ```bash
    git clone https://github.com/PongpreechaSuea/FaceDetection.git
    cd FaceDetection
    ```
2. Create and activate virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows use `venv\Scripts\activate`
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
Customize the `src/config.py` file as needed:
```python
MODEL = "./model/yolov8n-face.onnx"
IMAGE_PATH = "image_debug.jpg"

```


## Usage
### Starting the Server

Start the FastAPI server:

```cmd
python app.py || uvicorn app:app --host 127.0.0.1 --port 8000
```

## API Endpoints

### Upload and Detect Faces from Image

- Endpoint: /upload/payload/
- Method: POST
- Description: Upload an image and detect faces
- Request: multipart/form-data
    - file: Image for face detection
- Response:
    ```jsonCopy
    {
        "name_file": "example",
        "extension": "JPG",
        "file_input": "example.jpg",
        "type": "ImageFaceDetection",
        "count_img": 2,
        "data": [
            {
                "idx": 0,
                "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                "conf": 0.95
            },
            {
                "idx": 1,
                "bbox": {"x1": 500, "y1": 600, "x2": 700, "y2": 800},
                "conf": 0.88
            }
        ],
        "image": "base64_encoded_image_string"
    }
    ```

## Adjust Settings

- Endpoint: /seting
- Method: POST
- Description: Adjust confidence threshold and input size
- Request: Query parameters
    - conf_threshold: New confidence threshold value
    - input_size: New input size
- Response:
    ```jsonCopy
    {
        "status": 200,
        "action": "seting",
        "succeed": true,
        "project": "faceDetection",
        "model": "yolov8"
    }
    ```

## Stream Video from Camera

- Endpoint: /video_feed
- Method: GET
- Description: Stream video from camera with real-time face detection
- Response: multipart/x-mixed-replace stream

## Test Web Pages

- /upload: Page for uploading images
- /webcam: Page for opening camera and real-time face detection