import os
import logging
from typing import List, Dict, Any

import cv2
import numpy as np
import io
from typing import Optional

logger = logging.getLogger("process.utils")

MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# MobileNet-SSD (Caffe) model files
MOBILENET_PROTOTXT = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MOBILENET_CAFFEMODEL = os.path.join(MODELS_DIR, "mobilenet_iter_73000.caffemodel")

MOBILENET_PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
MOBILENET_CAFFEMODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

# Class labels for MobileNet SSD
MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]


def _download_file(url: str, dst_path: str):
    import requests

    if os.path.exists(dst_path):
        return
    logger.info(f"Downloading model from {url} -> {dst_path}")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with open(dst_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def ensure_mobilenet_model():
    try:
        _download_file(MOBILENET_PROTOTXT_URL, MOBILENET_PROTOTXT)
    except Exception:
        logger.warning("Failed to download prototxt; check network or provide manually")
    try:
        _download_file(MOBILENET_CAFFEMODEL_URL, MOBILENET_CAFFEMODEL)
    except Exception:
        logger.warning("Failed to download caffemodel; check network or provide manually")


def scene_keyframes(video_path: str, sample_rate: int = 5, diff_thresh: float = 30.0) -> List[Dict[str, Any]]:
    """Simple scene detection by frame-to-frame mean grayscale diff. Returns list of keyframes with frame index and timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    keyframes = []
    prev_gray = None
    frame_idx = 0
    last_keyframe_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            # first frame
            keyframes.append({"frame_idx": frame_idx, "timestamp": frame_idx / fps, "frame": frame.copy()})
            prev_gray = gray
            last_keyframe_idx = frame_idx
        else:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(diff))
            if mean_diff > diff_thresh and (frame_idx - last_keyframe_idx) > sample_rate:
                keyframes.append({"frame_idx": frame_idx, "timestamp": frame_idx / fps, "frame": frame.copy()})
                prev_gray = gray
                last_keyframe_idx = frame_idx
            else:
                # update prev for smoother detection
                prev_gray = gray
        frame_idx += 1
    cap.release()
    return keyframes


def load_mobilenet_net():
    ensure_mobilenet_model()
    if not (os.path.exists(MOBILENET_PROTOTXT) and os.path.exists(MOBILENET_CAFFEMODEL)):
        raise FileNotFoundError("MobileNet SSD model files not found in models directory")
    net = cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT, MOBILENET_CAFFEMODEL)
    return net


def detect_objects_on_image(net, image: np.ndarray, conf_thresh: float = 0.4):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_thresh:
            continue
        class_id = int(detections[0, 0, i, 1])
        label = MOBILENET_CLASSES[class_id] if class_id < len(MOBILENET_CLASSES) else str(class_id)
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        results.append({
            "label": label,
            "confidence": confidence,
            "bbox": [int(startX), int(startY), int(endX), int(endY)],
        })
    return results


_embedding_model = None


def load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            logger.exception("Failed to load embedding model; make sure sentence-transformers is installed")
            raise
    return _embedding_model


def summarize_detections(detections_by_frame: List[Dict[str, Any]]):
    # Aggregate by label
    summary = {}
    for det in detections_by_frame:
        frame_ts = det["timestamp"]
        for obj in det["objects"]:
            label = obj["label"]
            if label not in summary:
                summary[label] = {"label": label, "timestamps": [], "confidences": []}
            summary[label]["timestamps"].append(frame_ts)
            summary[label]["confidences"].append(obj["confidence"])
    # compute averages and embeddings
    embed_model = load_embedding_model()
    final = []
    for label, info in summary.items():
        avg_conf = float(np.mean(info["confidences"])) if info["confidences"] else 0.0
        timestamps = sorted(list(set([round(t, 3) for t in info["timestamps"]])))
        embedding = embed_model.encode(label).tolist()
        final.append({"label": label, "avg_confidence": avg_conf, "timestamps": timestamps, "embedding": embedding})
    return final


def serialize_embeddings(summary: List[Dict[str, Any]]) -> bytes:
    """Serialize embeddings from summary into a compressed npz bytes blob.

    summary: list of {label, avg_confidence, timestamps, embedding}
    returns: bytes
    """
    arrs = {}
    for item in summary:
        label = item.get("label")
        emb = np.array(item.get("embedding"), dtype=np.float32)
        # store as 1D array
        arrs[label] = emb
    bio = io.BytesIO()
    # use savez_compressed to store multiple arrays
    np.savez_compressed(bio, **arrs)
    bio.seek(0)
    return bio.read()


def deserialize_embeddings(blob: Optional[bytes]) -> Dict[str, np.ndarray]:
    """Deserialize embeddings blob (npz) into a dict label->np.ndarray.

    Returns empty dict if blob is falsy.
    """
    if not blob:
        return {}
    bio = io.BytesIO(blob)
    try:
        npz = np.load(bio, allow_pickle=False)
        result = {k: npz[k].astype(np.float32) for k in npz.files}
        return result
    except Exception:
        logger.exception("Failed to deserialize embeddings blob")
        return {}

