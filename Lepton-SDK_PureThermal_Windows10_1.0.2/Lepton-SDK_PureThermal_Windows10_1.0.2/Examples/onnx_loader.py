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

def predict_pattern_b_thermal(model, pil_img, frame_np,
                              coarse_imgsz=256, roi_imgsz=1024,
                              coarse_conf=0.01, roi_conf=0.08,
                              coarse_iou=0.35, roi_iou=0.5,
                              pad=0.6, max_rois=12,
                              thermal_k=2.5, thermal_min_area=20, thermal_max_rois=8):

    W, H = pil_img.size

    thermal_rois = get_thermal_rois(
        frame_np, k=thermal_k, min_area=thermal_min_area,
        max_rois=thermal_max_rois, pad=0.4
    )

    full_bgr = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR) if frame_np.ndim == 2 else frame_np
    c_boxes, c_scores, c_clss = model.predict(full_bgr, conf=coarse_conf, iou=coarse_iou)

    yolo_rois = []
    if len(c_boxes) > 0:
        order = np.argsort(-c_scores)
        yolo_rois = make_rois_from_boxes(c_boxes[order], W, H, pad=pad, max_rois=max_rois)

    rois = merge_rois(list(yolo_rois) + list(thermal_rois), iou_thr=0.3)

    all_boxes, all_scores, all_clss = [], [], []

    if len(c_boxes) > 0:
        all_boxes.append(c_boxes)
        all_scores.append(c_scores)
        all_clss.append(c_clss)

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
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    clss = np.concatenate(all_clss, axis=0)

    keep = model.nms_numpy(boxes, scores, iou_thresh=0.5)
    keep = keep[np.argsort(-scores[keep])]

    return boxes[keep], scores[keep], clss[keep]