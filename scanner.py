"""scanner.py — YOLO-based screen scanner for Buckshot Roulette overlay.

Runs inference through ONNX Runtime (CPU) instead of the full torch +
torchvision + ultralytics stack. Same best.pt weights (exported once to
best.onnx via `YOLO("best.pt").export(format="onnx", imgsz=1280)`), just a
far smaller/faster runtime for the packaged app. best.pt is kept untouched
for future retraining.

Pre-loads the model in a background thread so it is ready the moment
the user clicks SCAN ROUNDS.
"""

import os
import sys
import threading
import numpy as np
import mss
import onnxruntime as ort
from PIL import Image


def _resource_path(relative_path: str) -> str:
    """Resolve a bundled resource so it works both run from source and from
    a PyInstaller-frozen exe (--onefile extracts data next to sys._MEIPASS,
    not the process's current working directory)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


MODEL_PATH    = _resource_path("best.onnx")
IMG_SIZE      = 1280   # must match the imgsz used at export time (train.py used 1280)
CONFIDENCE    = 0.45
IOU_THRES     = 0.45
ITEM_SPLIT_Y  = 370    # screen px: above = enemy items, below = player items
MONITOR_INDEX = 1      # fallback only — see _primary_monitor()

# Class order baked into best.onnx's output tensor (from best.pt's model.names)
CLASS_NAMES = [
    "hp_bullet", "hp_point", "item_adrenaline", "item_beer", "item_ciggs",
    "item_cuffs", "item_glass", "item_inverter", "item_phone", "item_pills",
    "item_saw", "shell_blank", "shell_live",
]

# Maps class name → index in ITEMS_CONF (overlay slot order)
ITEM_IDX = {
    "item_glass":      0,
    "item_pills":      1,
    "item_phone":      2,
    "item_cuffs":      3,
    "item_adrenaline": 4,
    "item_saw":        5,
    "item_ciggs":      6,
    "item_beer":       7,
    "item_inverter":   8,
}


def _primary_monitor(sct):
    """Windows guarantees the primary monitor's origin is (0, 0) in virtual
    screen space; mss's monitor list order is not guaranteed to put it
    first, so search for it explicitly instead of assuming index 1."""
    for mon in sct.monitors[1:]:
        if mon["left"] == 0 and mon["top"] == 0:
            return mon
    return sct.monitors[MONITOR_INDEX]


def _letterbox(frame, size=IMG_SIZE, pad_value=114):
    """Resize + pad to a size x size square canvas, preserving aspect ratio
    (matches the square letterbox the model was trained/exported with)."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.array(Image.fromarray(frame).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def _preprocess(frame):
    canvas, scale, left, top = _letterbox(frame, IMG_SIZE)
    tensor = canvas.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None]  # HWC -> NCHW
    return tensor, scale, left, top


def _nms(boxes, confs, cls_ids, iou_thres):
    """Per-class greedy NMS. Returns indices to keep."""
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        same_cls = cls_ids[rest] == cls_ids[i]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[~(same_cls & (iou > iou_thres))]
    return keep


def _postprocess(raw_output, scale, left, top, conf_thres=CONFIDENCE, iou_thres=IOU_THRES):
    """raw_output: (1, 4+num_classes, num_boxes) -> list of
    (x1, y1, x2, y2, conf, class_name) in ORIGINAL frame coordinates."""
    pred = raw_output[0].T  # (num_boxes, 4+num_classes)
    boxes_cxcywh = pred[:, :4]
    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(axis=1)
    confs = scores_all.max(axis=1)

    mask = confs > conf_thres
    boxes_cxcywh, confs, cls_ids = boxes_cxcywh[mask], confs[mask], cls_ids[mask]
    if boxes_cxcywh.shape[0] == 0:
        return []

    cx, cy, w, h = boxes_cxcywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    keep_idx = _nms(boxes, confs, cls_ids, iou_thres)
    boxes, confs, cls_ids = boxes[keep_idx], confs[keep_idx], cls_ids[keep_idx]

    # Undo the letterbox to get back to original frame coordinates
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale

    return [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), CLASS_NAMES[int(k)])
        for b, c, k in zip(boxes, confs, cls_ids)
    ]


class Scanner:
    def __init__(self):
        self._session    = None
        self._input_name = None
        self._ready       = False
        self._loading     = True
        threading.Thread(target=self._preload, daemon=True).start()

    # ── Internal ──────────────────────────────────────────────────────────────
    def _preload(self):
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Warm-up pass so first real inference is instant
        dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        session.run(None, {input_name: dummy})

        self._session    = session
        self._input_name = input_name
        self._loading    = False
        self._ready      = True

    def _detect(self):
        """Capture the primary monitor and return decoded detections in
        screen coordinates, or None if the model isn't ready yet."""
        if not self._ready:
            return None

        with mss.mss() as sct:
            mon = _primary_monitor(sct)
            frame = np.array(sct.grab(mon))[..., :3]

        tensor, scale, left, top = _preprocess(frame)
        raw = self._session.run(None, {self._input_name: tensor})[0]
        return _postprocess(raw, scale, left, top)

    # ── Public ────────────────────────────────────────────────────────────────
    @property
    def is_ready(self):
        return self._ready

    @property
    def is_loading(self):
        return self._loading

    def scan(self):
        """Detect live/blank shell counts only.

        Returns dict:
            live  — int, shell_live count
            blank — int, shell_blank count
        Or None if model not ready.
        """
        detections = self._detect()
        if detections is None:
            return None

        live = blank = 0
        for *_, label in detections:
            if label == "shell_live":
                live += 1
            elif label == "shell_blank":
                blank += 1
        return {"live": live, "blank": blank}

    def scan_items(self):
        """Detect items by Y position.

        Returns dict:
            player — list[int] len 9, count per item slot index
            enemy  — list[int] len 9, count per item slot index
        Or None if model not ready.
        """
        detections = self._detect()
        if detections is None:
            return None

        player = [0] * 9
        enemy  = [0] * 9
        for x1, y1, x2, y2, conf, label in detections:
            idx = ITEM_IDX.get(label)
            if idx is None:
                continue
            cy = (y1 + y2) / 2
            if cy < ITEM_SPLIT_Y:
                enemy[idx]  += 1
            else:
                player[idx] += 1
        return {"player": player, "enemy": enemy}
