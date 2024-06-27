import cv2
import numpy as np
import os
import math
import base64
from src.config import MODEL

class YOLOv8FaceDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = 0.8, input_size: int = 480):
        self.conf_threshold = conf_threshold  # ค่าความเชื่อมั่นขั้นต่ำในการยอมรับการตรวจจับใบหน้า
        self.input_size = input_size  # ขนาดของภาพที่ใช้เป็นอินพุตให้โมเดล
        self.net = cv2.dnn.readNet(model_path)  # โหลดโมเดล
        self.init_params()  # กำหนดค่าพารามิเตอร์อื่นๆ

    def init_params(self):
        self.class_names = ['face']  # ชื่อคลาสที่โมเดลตรวจจับ
        self.num_classes = len(self.class_names)
        self.iou_threshold = 0.58  # ค่า IoU ขั้นต่ำในการยอมรับการตรวจจับใบหน้า
        self.reg_max = 16
        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)  # ค่าสเตรดที่ใช้ในการตรวจจับใบหน้า
        self.feats_hw = [(math.ceil(self.input_size / stride), math.ceil(self.input_size / stride)) for stride in self.strides]
        self.anchors = self.make_anchors(self.feats_hw)  # สร้างจุดสำหรับกำหนดตำแหน่งใบหน้า

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset
            y = np.arange(0, h) + grid_cell_offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        return x_exp / x_sum

    def resize_image(self, image, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_size, self.input_size
        if keep_ratio and image.shape[0] != image.shape[1]:
            hw_scale = image.shape[0] / image.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_size, int(self.input_size / hw_scale)
                img = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_size - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_size - neww - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                newh, neww = int(self.input_size * hw_scale), self.input_size
                img = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_size - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_size - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, image):
        if len(image.shape) != 3 or image.shape[2] != 3:
            print("Invalid image shape. Expected shape: (height, width, 3)")
            return np.array([]), np.array([]), np.array([])
             
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = image.shape[0] / newh, image.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0
        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        det_bboxes, det_conf, det_landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        return det_bboxes, det_conf, det_landmarks
    
    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_size / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))

            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_size, self.input_size)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        try:
            indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold).flatten()
        except:
            return np.array([]), np.array([]), np.array([])

        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, landmarks
        else:
            print('Nothing detected')
            return np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def draw_detections(self, image, boxes):
        for box in boxes:
            x, y, w, h = box.astype(int)
            cv2.rectangle(image, (x, y), (int(x + w), int(y + h)), (0, 0, 255), thickness=2)
        return image



class YOLOFaceImageDetector:
    def __init__(self, conf_threshold: float = 0.3, input_size: int = 480):
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.save_json = True
        self.labels_detect = False
        self.detector = YOLOv8FaceDetector(MODEL, conf_threshold=self.conf_threshold, input_size=self.input_size)
        self.space = 20 

    def seting(self):
        self.detector.conf_threshold = self.conf_threshold
        self.detector.input_size = self.input_size
        self.detector.init_params()

    def run(self, image_data, filename):
        json_data = {}
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        name, ext = os.path.splitext(filename)
        ext = ext[1:].upper()

        boxes, confidences, _ = self.detector.detect(image)

        if len(boxes) == 0:
            json_data["name_file"] = name
            json_data["extension"] = ext
            json_data["file_input"] = filename
            json_data["type"] = "ImageFaceDetection"
            json_data["error"] = "No faces detected"

        else:
            image_draw = self.detector.draw_detections(image,boxes)
            _, buffer = cv2.imencode('.jpg', image_draw)
            encoded_image = base64.b64encode(buffer).decode('utf-8')


            data = []
            for idx, (box, conf) in enumerate(zip(boxes, confidences)):
                x, y, w, h = box.astype(int)
                # face_capture = image[y - self.space:y + h + self.space, x - self.space:x + w + self.space]

                data.append({
                    "idx": idx,
                    "bbox": {"x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h)},
                    "conf": float(conf),
                    # "face_capture": face_capture.tolist(),
                })

            json_data["name_file"] = name
            json_data["extension"] = ext
            json_data["file_input"] = filename
            json_data["type"] = "ImageFaceDetection"
            json_data["count_img"] = len(confidences)
            json_data["data"] = data
            json_data["image"] = encoded_image

        return json_data

    def webcam(self, image_data):
        boxes, _, _ = self.detector.detect(image_data)
        image_draw = self.detector.draw_detections(image_data, boxes)
        return image_draw
    
    def webcam_upload(self, image_data):
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        boxes, _, _ = self.detector.detect(image)
        image_draw = self.detector.draw_detections(image, boxes)
        _, buffer = cv2.imencode('.jpg', image_draw)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return {"image" : encoded_image}