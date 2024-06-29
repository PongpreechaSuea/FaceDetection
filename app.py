from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from src.detect import YOLOFaceImageDetector
from src.camera_multi import Camera
from src.config import *
from PIL import Image
import cv2
import io
import asyncio

app = FastAPI(
    title="Face Detection API",
    description="An advanced face detection system using YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ปรับตามความเหมาะสม
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

faceDetection = YOLOFaceImageDetector()
faceDetection.seting()

async def verify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    return file

@app.get("/", tags=["Info"])
async def root():
    return {
        "status": 200,
        "message": "Welcome to the Face Detection API!",
        "model": "YOLOv8",
        "features": [
            "Image upload detection",
            "Real-time webcam detection",
            "Adjustable confidence threshold",
            "Customizable input size"
        ]
    }

@app.get("/detail", tags=["Settings"])
async def detail():
    return {
        "status": 200,
        "conf_threshold": faceDetection.conf_threshold,
        "input_size": faceDetection.input_size,
        "model": "YOLOv8"
    }

@app.post("/seting", tags=["Settings"])
async def seting(conf_threshold: float = 0.5, input_size: int = 480):
    if conf_threshold > 1 or conf_threshold < 0:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0 and 1")
    if input_size <= 0:
        raise HTTPException(status_code=400, detail="Input size must be positive")
    
    faceDetection.conf_threshold = conf_threshold
    faceDetection.input_size = input_size
    faceDetection.seting()
    return {"status": 200, "message": "Settings updated successfully"}

@app.post("/upload/img/", tags=["Detection"])
async def upload_image(file: UploadFile = Depends(verify_image)):
    contents = await file.read()
    result = await asyncio.to_thread(faceDetection.webcam_upload, contents)
    return JSONResponse(content=result)

@app.post("/upload/payload/", tags=["Detection"])
async def upload_image_payload(file: UploadFile = Depends(verify_image)):
    contents = await file.read()
    result = await asyncio.to_thread(faceDetection.run, contents, file.filename)
    return JSONResponse(content=result, status_code=400 if "error" in result else 200)

@app.post("/upload/size", tags=["Utility"])
async def size_img(file: UploadFile = Depends(verify_image)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    return {"width": img.width, "height": img.height}

@app.get("/upload", tags=["UI"])
async def index_uploadFlieImg(request: Request):
    return templates.TemplateResponse("uploadFlieImg.html", {"request": request})

@app.get("/webcam", tags=["UI"])
async def index_openCamera(request: Request):
    return templates.TemplateResponse("openCamera.html", {"request": request})

async def gen(camera):
    try:
        while True:
            frame, img = await asyncio.to_thread(camera.get_frame)
            try:
                img = await asyncio.to_thread(faceDetection.webcam, img)
                _, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
            except Exception as e:
                print(f"Error processing frame: {e}")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    finally:
        camera.release()

@app.get("/video_feed", tags=["Streaming"])
async def video_feed():
    camera = Camera()
    return StreamingResponse(gen(camera), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print(f"Upload UI: {BASE_URL}/upload")
    print(f"Webcam UI: {BASE_URL}/webcam")
    print(f"Swagger UI: {SWAGGER}")
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)