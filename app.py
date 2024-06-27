from fastapi import FastAPI, File, UploadFile, Request
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from src.detect_yolo import YOLOFaceImageDetector

from PIL import Image
import uvicorn

import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'],
)

faceDetection = YOLOFaceImageDetector()
faceDetection.seting()

# ========================================================================= 

# def info():
#     return {
#         "status": 200,
#         "action" : "default",
#         "succeed" : True,
#         "project" : "face_detection",
#         "model" : "yolov8"
#     }


@app.get("/")
async def root():
    return {
        "status": 200,
        "action" : "default",
        "succeed" : True,
        "project" : "face_detection",
        "model" : "yolov8"
    }

@app.get("/detail")
async def detail():
    return {
        "status": 200,
        "action" : "detail",
        "succeed" : True,
        "conf_threshold" : faceDetection.conf_threshold,
        "input_size" : faceDetection.input_size,
        "model" : "yolov8"
    }


@app.post("/seting")
async def seting(conf_threshold: float = 0.5, input_size: int = 320):
    try:
        if conf_threshold > 1:
            raise HTTPException(status_code=400, detail="conf_threshold must be less than or equal to 1")
        
        faceDetection.conf_threshold = conf_threshold
        faceDetection.input_size = input_size
        faceDetection.seting()
        return {
            "status": 200,
            "action" : "seting",
            "succeed" : True,
            "action": "seting",
            "project": "faceDetection",
            "model": "yolov8"
        }
    except Exception as e:
        return {
            "status": int(str(e).split(":")[0]),
            "action" : "seting",
            "succeed" : False,
            "error": str(e).split(":")[1],
        }


@app.post("/upload/img/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = faceDetection.webcam_upload(contents)
    return JSONResponse(content=result)
    

@app.post("/upload/payload/")
async def upload_image_payload(file: UploadFile = File(...)):
    contents = await file.read()
    result = faceDetection.run(contents, file.filename)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    else:
        return JSONResponse(content=result)

@app.post("/upload/size")
async def size_img(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    width, height = img.size
    return {"width": width, "height": height}


# ========================================================================= Use
 
@app.get("/upload")
async def index_uploadFlieImg(request: Request):
    return templates.TemplateResponse("uploadFlieImg.html", {"request": request})

@app.get("/webcam")
async def index_openCamera(request: Request):
    return templates.TemplateResponse("openCamera.html", {"request": request})


# ========================================================================= เปิดกล้องในเครื่อง


from src.camera_multi import Camera
import cv2

def gen(camera):
    while True:
        frame , img = camera.get_frame()
        try:
            img = faceDetection.webcam(img)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
        except:
            pass
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen(Camera()), media_type="multipart/x-mixed-replace; boundary=frame")


# ========================================================================= 

if __name__ == "__main__":
#   uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
  uvicorn.run("app:app", host="127.0.0.1", port=8000)

