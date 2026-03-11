import cv2
import numpy as np
import torch, os, glob, json
from torchvision.ops import nms
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmengine.dataset import pseudo_collate
from mmdet.structures import DetDataSample
from mmcv.transforms import Compose

def get_thermal_rois(frame_np,
                     k=2.5,            # threshold = mean + k*std (tune 1.5–4)
                     min_area=20,       # pixels in 160x120 space (tune)
                     max_rois=8,
                     pad=0.4):
    """
    Returns list of ROIs (x1,y1,x2,y2) in image pixel coords.
    Designed for low-res thermal where targets are warmer than background.
    """
    if frame_np.ndim == 3:
        gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_np.copy()

    gray = gray.astype(np.float32)

    # light denoise to reduce speckle
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thr = np.percentile(frame_np, 98.5)

    mask = (gray_blur > thr).astype(np.uint8) * 255

    # morphology to connect blobs + remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape[:2]
    rois = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue

        # pad ROI
        px, py = pad * w, pad * h
        x1 = max(0, int(np.floor(x - px)))
        y1 = max(0, int(np.floor(y - py)))
        x2 = min(W, int(np.ceil(x + w + px)))
        y2 = min(H, int(np.ceil(y + h + py)))
        rois.append((x1, y1, x2, y2))

    # keep largest blobs (by area)
    rois = sorted(rois, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)[:max_rois]
    return rois

def merge_rois(rois, iou_thr=0.3):
    """Simple greedy merge of highly-overlapping ROIs."""
    if not rois:
        return []

    boxes = np.array(rois, dtype=np.float32)
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    order = np.argsort(-areas)

    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(tuple(boxes[i].astype(int)))
        if order.size == 1:
            break
        rest = boxes[order[1:]]

        # IoU with kept box
        xx1 = np.maximum(boxes[i,0], rest[:,0])
        yy1 = np.maximum(boxes[i,1], rest[:,1])
        xx2 = np.minimum(boxes[i,2], rest[:,2])
        yy2 = np.minimum(boxes[i,3], rest[:,3])

        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        union = areas[i] + ((rest[:,2]-rest[:,0])*(rest[:,3]-rest[:,1])) - inter
        iou = inter / (union + 1e-6)

        order = order[1:][iou < iou_thr]

    return kept

def make_rois_from_boxes(boxes_xyxy, img_w, img_h, pad=0.25, max_rois=6, min_side=12):
    rois = []
    for (x1,y1,x2,y2) in boxes_xyxy:
        w = x2 - x1
        h = y2 - y1
        # pad
        px = pad * w
        py = pad * h
        rx1 = max(0, int(np.floor(x1 - px)))
        ry1 = max(0, int(np.floor(y1 - py)))
        rx2 = min(img_w, int(np.ceil(x2 + px)))
        ry2 = min(img_h, int(np.ceil(y2 + py)))
        if (rx2 - rx1) < min_side or (ry2 - ry1) < min_side:
            continue
        rois.append((rx1, ry1, rx2, ry2))

    # simple cap; you can also merge overlapping ROIs if needed
    return rois[:max_rois]

@torch.no_grad()
def predict_pattern_b_thermal(model, pil_img, frame_np,
                              coarse_imgsz=640, roi_imgsz=1024,
                              coarse_conf=0.01, roi_conf=0.08,
                              coarse_iou=0.35, roi_iou=0.5,
                              pad=0.6, max_rois=12, max_det_coarse=500,
                              thermal_k=2.5, thermal_min_area=20, thermal_max_rois=8):
    W, H = pil_img.size

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    # --- NEW: thermal ROIs ---
    thermal_rois = get_thermal_rois(
        frame_np, k=thermal_k, min_area=thermal_min_area,
        max_rois=thermal_max_rois, pad=0.4
    )

    # --- Coarse YOLO pass (proposal-biased) ---
    coarse = model.predict(
        pil_img, imgsz=coarse_imgsz, conf=coarse_conf,
        iou=coarse_iou, max_det=max_det_coarse,
        device=DEVICE, verbose=False
    )[0]

    yolo_rois = []
    if coarse.boxes is not None and len(coarse.boxes) > 0:
        c_xyxy = coarse.boxes.xyxy.cpu().numpy()
        c_conf = coarse.boxes.conf.cpu().numpy()
        order = np.argsort(-c_conf)
        yolo_rois = make_rois_from_boxes(c_xyxy[order], W, H, pad=pad, max_rois=max_rois)

    # Combine ROI sources
    rois = merge_rois(list(yolo_rois) + list(thermal_rois), iou_thr=0.3)

    # Keep coarse detections (if any)
    all_boxes, all_scores, all_clss = [], [], []
    if coarse.boxes is not None and len(coarse.boxes) > 0:
        all_boxes.append(coarse.boxes.xyxy.cpu().numpy())
        all_scores.append(coarse.boxes.conf.cpu().numpy())
        all_clss.append(coarse.boxes.cls.cpu().numpy().astype(int))

    # --- ROI zoom pass ---
    for (x1, y1, x2, y2) in rois:
        crop = pil_img.crop((x1, y1, x2, y2))
        r = model.predict(
            crop, imgsz=roi_imgsz, conf=roi_conf,
            iou=roi_iou, max_det=200,
            device=DEVICE, verbose=False
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        k = r.boxes.cls.cpu().numpy().astype(int)

        b[:, [0, 2]] += x1
        b[:, [1, 3]] += y1

        all_boxes.append(b); all_scores.append(s); all_clss.append(k)

    if not all_boxes:
        return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    clss = np.concatenate(all_clss, axis=0)

    keep = nms(torch.tensor(boxes, dtype=torch.float32),
               torch.tensor(scores, dtype=torch.float32),
               iou_threshold=0.5).cpu().numpy()

    keep = keep[np.argsort(-scores[keep])]
    return boxes[keep], scores[keep], clss[keep]

def detect_deer_in_rois(model, frame8, rois, roi_imgsz=640, roi_conf=0.005, roi_iou=0.5):
    """
    frame8: 8-bit image used for inference, shape (H,W) or (H,W,3)
    rois: list of (x1,y1,x2,y2)
    returns: boxes, scores, classes in full-image coordinates
    """
    if frame8.ndim == 2:
        rgb = cv2.cvtColor(frame8, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame8, cv2.COLOR_BGR2RGB)

    # pil_full = Image.fromarray(rgb)

    all_boxes, all_scores, all_clss = [], [], []

    for (x1, y1, x2, y2) in rois:
        # crop = pil_full.crop((x1, y1, x2, y2))
        crop = rgb[y1:y2, x1:x2]
        # result = model.predict(
        #     crop,
        #     imgsz=roi_imgsz,
        #     conf=roi_conf,
        #     iou=roi_iou,
        #     verbose=False
        # )[0]

        # if result.boxes is None or len(result.boxes) == 0:
        #     continue
        if crop.size == 0:
            continue

        data = dict(
            img=crop,
            img_id=0
        )
        test_pipeline = model.cfg.test_dataloader.dataset.pipeline
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(test_pipeline)
        data = test_pipeline(data)
        data = pseudo_collate([data])

        with torch.no_grad():
            result = model.test_step(data)[0]

        pred = result.pred_instances

        if len(pred) == 0:
            continue

        # boxes = result.boxes.xyxy.cpu().numpy()
        # scores = result.boxes.conf.cpu().numpy()
        # clss = result.boxes.cls.cpu().numpy().astype(int)

        boxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        clss = pred.labels.detach().cpu().numpy().astype(int)

        keep = scores >= roi_conf
        boxes, scores, clss = boxes[keep], scores[keep], clss[keep]

        if len(boxes) == 0:
            continue
        # move crop-local boxes back into full-image coords
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_clss.append(clss)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    return (
        np.concatenate(all_boxes, axis=0),
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_clss, axis=0),
    )

def detect_full_frame(model, frame8, conf_thr=0.01):
    if frame8.ndim == 2:
        full_rgb = cv2.cvtColor(frame8, cv2.COLOR_GRAY2RGB)
    else:
        full_rgb = cv2.cvtColor(frame8, cv2.COLOR_BGR2RGB)
        
    test_pipeline = model.cfg.test_dataloader.dataset.pipeline
    test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline)
    data = dict(img=np.asarray(full_rgb), img_id=0)
    data = test_pipeline(data)
    data = pseudo_collate([data])

    with torch.no_grad():
        result = model.test_step(data)[0]

    pred = result.pred_instances
    if len(pred) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    boxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    clss = pred.labels.detach().cpu().numpy().astype(int)

    keep = scores >= conf_thr
    return boxes[keep], scores[keep], clss[keep]