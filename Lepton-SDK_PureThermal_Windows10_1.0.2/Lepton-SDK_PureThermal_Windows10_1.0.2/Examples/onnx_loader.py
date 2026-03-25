class ONNXDetector:
    def __init__(self, onnx_path, imgsz=640, providers=None):
        self.imgsz = imgsz

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def letterbox(self, img, new_shape=640, color=(114, 114, 114)):
        h, w = img.shape[:2]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        return img, r, (dw, dh)

    def preprocess(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_lb, ratio, (dw, dh) = self.letterbox(img, self.imgsz)

        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))   # HWC -> CHW
        x = np.expand_dims(x, axis=0)    # CHW -> NCHW

        return x, ratio, dw, dh

    def xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def nms_numpy(self, boxes, scores, iou_thresh=0.5):
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            union = areas[i] + areas[order[1:]] - inter + 1e-9
            iou = inter / union

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    def postprocess(self, pred, orig_shape, ratio, dw, dh, conf=0.25, iou=0.45):
        h0, w0 = orig_shape[:2]

        # squeeze batch
        if pred.ndim == 3:
            pred = pred[0]

        # Ultralytics ONNX commonly gives either:
        # (84, N) or (N, 84)
        if pred.shape[0] < pred.shape[1] and pred.shape[0] in (6, 7, 84, 85, 86):
            pred = pred.T

        if pred.shape[1] < 6:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        boxes = pred[:, :4]
        cls_scores = pred[:, 4:]

        clss = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(cls_scores)), clss]

        keep = scores >= conf
        boxes = boxes[keep]
        scores = scores[keep]
        clss = clss[keep]

        if len(boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        boxes = self.xywh2xyxy(boxes)

        # undo letterbox
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :4] /= ratio

        boxes[:, 0] = boxes[:, 0].clip(0, w0 - 1)
        boxes[:, 1] = boxes[:, 1].clip(0, h0 - 1)
        boxes[:, 2] = boxes[:, 2].clip(0, w0 - 1)
        boxes[:, 3] = boxes[:, 3].clip(0, h0 - 1)

        final_keep = self.nms_numpy(boxes, scores, iou_thresh=iou)

        return (
            boxes[final_keep].astype(np.float32),
            scores[final_keep].astype(np.float32),
            clss[final_keep].astype(np.int32),
        )

    def predict(self, img, conf=0.25, iou=0.45):
        x, ratio, dw, dh = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: x})

        pred = outputs[0]
        return self.postprocess(pred, img.shape, ratio, dw, dh, conf=conf, iou=iou)


def predict_pattern_b_thermal(model, pil_img, frame_np,
                              coarse_imgsz=256, roi_imgsz=1024,
                              coarse_conf=0.01, roi_conf=0.08,
                              coarse_iou=0.35, roi_iou=0.5,
                              pad=0.6, max_rois=12, max_det_coarse=500,
                              thermal_k=2.5, thermal_min_area=20, thermal_max_rois=8):

    W, H = pil_img.size

    thermal_rois = get_thermal_rois(
        frame_np, k=thermal_k, min_area=thermal_min_area,
        max_rois=thermal_max_rois, pad=0.4
    )

    # coarse full-frame pass
    full_bgr = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR) if frame_np.ndim == 2 else frame_np
    c_boxes, c_scores, c_clss = model.predict(full_bgr, conf=coarse_conf, iou=coarse_iou)

    yolo_rois = []
    if len(c_boxes) > 0:
        order = np.argsort(-c_scores)
        yolo_rois = make_rois_from_boxes(c_boxes[order], W, H, pad=pad, max_rois=max_rois)

    rois = merge_rois(list(yolo_rois) + list(thermal_rois), iou_thr=0.3)

    all_boxes, all_scores, all_clss = [], [], []

    # keep coarse detections
    if len(c_boxes) > 0:
        all_boxes.append(c_boxes)
        all_scores.append(c_scores)
        all_clss.append(c_clss)

    # ROI zoom pass
    for (x1, y1, x2, y2) in rois:
        crop = frame_np[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop

        b, s, k = model.predict(crop_bgr, conf=roi_conf, iou=roi_iou)

        if len(b) == 0:
            continue

        b[:, [0, 2]] += x1
        b[:, [1, 3]] += y1

        all_boxes.append(b)
        all_scores.append(s)
        all_clss.append(k)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    clss = np.concatenate(all_clss, axis=0)

    keep = model.nms_numpy(boxes, scores, iou_thresh=0.5)
    keep = keep[np.argsort(-scores[keep])]

    return boxes[keep], scores[keep], clss[keep]