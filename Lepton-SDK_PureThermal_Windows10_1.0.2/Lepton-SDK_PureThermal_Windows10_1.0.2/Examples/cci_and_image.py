from import_clr import *

clr.AddReference("ManagedIR16Filters")

from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent
import numpy as np
import time
import cv2
import pythoncom
from PIL import Image
from ultralytics import YOLO
from utils import predict_pattern_b_thermal, detect_deer_in_rois, get_thermal_rois, merge_rois, detect_full_frame
from mmdet.apis import init_detector, inference_detector
from mmcv.transforms import Compose

pythoncom.CoInitialize()

lep, = (dev.Open() for dev in CCI.GetDevices())
print("Camera uptime:", lep.sys.GetCameraUpTime())

# Shared buffer updated by callback
numpyArr = None

def getFrameRaw(arr, width, height):
    global numpyArr
    # arr is an iterable of uint16 values
    numpyArr = np.fromiter(arr, dtype=np.uint16).reshape(height, width)


capture = IR16Capture()
capture.vignette = True
capture.noise_filter = True
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(getFrameRaw))
capture.RunGraph()

win = "Lepton Live"
win2 = "Lepton Resized"

# model = YOLO("best.pt")

# model.export(format="engine", half=True, imgsz=640)
CONFIG = r"C:\Users\Johnny\Desktop\deer_detector\mmyolo\configs\yolov8_test.py"
CHECKPOINT = r"C:\Users\Johnny\Desktop\deer_detector\mmyolo\work_dirs\yolov8_test\best_coco_bbox_mAP_epoch_200.pth"
DEVICE = "cuda:0"   # or "cpu"

model = init_detector(CONFIG, CHECKPOINT, device=DEVICE)

DEER_CLASS_ID = 0          
ALERT_CONF = 0.01         
ALERT_COOLDOWN = 10.0      
last_alert_time = 0.0

try:
    while True:
        if numpyArr is None:
            time.sleep(0.01)
            continue

        frame16 = numpyArr  # (H, W) uint16

        # Optional: upsample for nicer viewing
        up = cv2.resize(frame16, (240, 240), interpolation=cv2.INTER_LINEAR)

        # Normalize 16-bit -> 8-bit for display
        img = cv2.normalize(up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img8 = cv2.medianBlur(img, 3) 
        col_mean = np.mean(img8, axis=0)
        img_destriped = img8 - col_mean + np.mean(col_mean)
        img_destriped = np.clip(img_destriped, 0, 255).astype(np.uint8)
        # # Optional: apply a colormap (comment out if you want pure grayscale)
        # disp = cv2.applyColorMap(disp8, cv2.COLORMAP_INFERNO)
        # disp = disp8  # use this instead for grayscale
        
        rois = get_thermal_rois(img_destriped, min_area=30, max_rois=8, pad=0.3)
        rois = merge_rois(rois, iou_thr=0.3)

        # 2. run detector only on those ROIs
        boxes, scores, clss = detect_deer_in_rois(model, img_destriped, rois)

        vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        deer_found = False
        best_score = 0.0

        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, clss):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if cls == DEER_CLASS_ID:
                color = (0, 255, 0)
                deer_found = deer_found or (score >= ALERT_CONF)
                best_score = max(best_score, float(score))
                label = f"deer {score:.2f}"
            else:
                color = (0, 255, 255)
                label = f"{cls} {score:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        for (x1, y1, x2, y2) in rois:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # 4. alert with cooldown
        now = time.time()
        if deer_found and (now - last_alert_time > ALERT_COOLDOWN):
            print(f"Deer detected with confidence {best_score:.2f}")
            last_alert_time = now

        cv2.imshow("detections", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

finally:
    capture.StopGraph()
    capture.Dispose()
    cv2.destroyAllWindows()
