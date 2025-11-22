import os
import logging
from typing import List, Dict, Any

import cv2
import numpy as np
import io
import json
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

_whisper_model = None
_image_embedding_model = None

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


def scene_keyframes(video_path: str, sample_rate: int = 5, diff_thresh: float = 30.0, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    keyframes = []
    prev_gray = None
    frame_idx = 0
    last_keyframe_idx = 0
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            kf_dict = {"frame_idx": frame_idx, "timestamp": frame_idx / fps}
            
            if output_dir:
                kf_path = os.path.join(output_dir, f"kf_{frame_idx}.jpg")
                cv2.imwrite(kf_path, frame)
                kf_dict["frame_path"] = kf_path
            else:
                kf_dict["frame"] = frame.copy()
            
            keyframes.append(kf_dict)
            prev_gray = gray
            last_keyframe_idx = frame_idx
        else:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(np.mean(diff))
            if mean_diff > diff_thresh and (frame_idx - last_keyframe_idx) > sample_rate:
                kf_dict = {"frame_idx": frame_idx, "timestamp": frame_idx / fps}
                
                if output_dir:
                    kf_path = os.path.join(output_dir, f"kf_{frame_idx}.jpg")
                    cv2.imwrite(kf_path, frame)
                    kf_dict["frame_path"] = kf_path
                else:
                    kf_dict["frame"] = frame.copy()
                
                keyframes.append(kf_dict)
                prev_gray = gray
                last_keyframe_idx = frame_idx
            else:
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


def detect_objects_on_image(net, image: np.ndarray, conf_thresh: float = 0.7, draw_boxes: bool = False):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    results = []
    annotated_image = image.copy() if draw_boxes else None
    
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_thresh:
            continue
        class_id = int(detections[0, 0, i, 1])
        label = MOBILENET_CLASSES[class_id] if class_id < len(MOBILENET_CLASSES) else str(class_id)
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        if draw_boxes and annotated_image is not None:
            cv2.rectangle(annotated_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label_text = f"{label} ({confidence:.2f})"
            (text_width, text_height) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_rect_top = startY - text_height - 8
            text_rect_bottom = startY
            inside = False
            if text_rect_top < 0:
                text_rect_top = startY + 4
                text_rect_bottom = startY + text_height + 12
                inside = True

            cv2.rectangle(annotated_image, (startX, int(text_rect_top)),
                          (startX + text_width, int(text_rect_bottom)), (0, 255, 0), -1)

            if inside:
                text_org_y = int(text_rect_bottom - 4)
            else:
                text_org_y = int(text_rect_bottom - 4)

            cv2.putText(annotated_image, label_text, (startX, text_org_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        results.append({
            "label": label,
            "confidence": confidence,
            "bbox": [int(startX), int(startY), int(endX), int(endY)],
        })
    
    return (annotated_image if draw_boxes else image, results)


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
    summary = {}
    for det in detections_by_frame:
        frame_ts = det["timestamp"]
        for obj in det["objects"]:
            label = obj["label"]
            if label not in summary:
                summary[label] = {"label": label, "timestamps": [], "confidences": []}
            summary[label]["timestamps"].append(frame_ts)
            summary[label]["confidences"].append(obj["confidence"])
    embed_model = load_embedding_model()
    final = []
    for label, info in summary.items():
        avg_conf = float(np.mean(info["confidences"])) if info["confidences"] else 0.0
        timestamps = sorted(list(set([round(t, 3) for t in info["timestamps"]])))
        embedding = embed_model.encode(label).tolist()
        final.append({"label": label, "avg_confidence": avg_conf, "timestamps": timestamps, "embedding": embedding})
    return final


def serialize_embeddings(summary: List[Dict[str, Any]]) -> bytes:
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



def load_whisper_model(model_name: str = "openai/whisper-tiny"):
    global _whisper_model
    if _whisper_model is None:
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(model_name)

            _whisper_model = {"model": model, "processor": processor, "device": device, "torch_dtype": torch_dtype}
            logger.info(f"Loaded Whisper model: {model_name} on device: {device}")
        except Exception:
            logger.exception("Failed to load Whisper model")
            raise
    return _whisper_model


def load_audio(audio_path: str) -> tuple:
    try:
        import librosa

        audio_data, sr = librosa.load(audio_path, sr=None)
        logger.info(f"Loaded audio from {audio_path}: sr={sr}, duration={len(audio_data)/sr:.2f}s")
        return audio_data, sr
    except Exception:
        logger.exception(f"Failed to load audio file: {audio_path}")
        raise


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    try:
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")

        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        audio_data = audio_data - np.mean(audio_data)
        std = np.std(audio_data)
        if std > 0:
            audio_data = audio_data / std

        logger.info(f"Normalized audio: mean={np.mean(audio_data):.6f}, std={np.std(audio_data):.6f}")
        return audio_data
    except Exception:
        logger.exception("Failed to normalize audio")
        raise


def preprocess_audio(audio_path: str, target_sr: int = 16000) -> tuple:
    try:
        import librosa

        audio_data, original_sr = load_audio(audio_path)

        if original_sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
            logger.info(f"Resampled audio from {original_sr}Hz to {target_sr}Hz")

        audio_data = normalize_audio(audio_data)

        return audio_data, original_sr, target_sr
    except Exception:
        logger.exception("Failed to preprocess audio")
        raise


def transcribe_audio(audio_path: str, model_name: str = "openai/whisper-tiny") -> Dict[str, Any]:
    try:
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            stride_length_s=(4, 2),
        )

        audio_data, original_sr, target_sr = preprocess_audio(audio_path, target_sr=16000)

        result = pipe(audio_data, return_timestamps=True, language='en')

        full_text = result.get("text", "")

        segments = []
        if "chunks" in result:
            for idx, chunk in enumerate(result["chunks"]):
                # extract confidence/score if available in chunk metadata
                conf = None
                if isinstance(chunk, dict):
                    conf = chunk.get("confidence") or chunk.get("score") or chunk.get("avg_logprob")

                segment = {
                    "id": idx,
                    "start": chunk.get("timestamp", [0, 0])[0] if isinstance(chunk.get("timestamp"), (list, tuple)) else 0,
                    "end": chunk.get("timestamp", [0, 0])[1] if isinstance(chunk.get("timestamp"), (list, tuple)) else 0,
                    "text": chunk.get("text", ""),
                    "confidence": float(conf) if conf is not None else None,
                }
                segments.append(segment)
        else:
            segments = [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": len(audio_data) / target_sr,
                    "text": full_text,
                    "confidence": None,
                }
            ]

        logger.info(f"Transcribed audio: {len(segments)} segments, full_text={full_text[:100]}")

        return {"text": full_text, "segments": segments}
    except Exception:
        logger.exception("Failed to transcribe audio")
        raise


def generate_text_embeddings(text_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        embed_model = load_embedding_model()
        enriched_segments = []

        for segment in text_segments:
            text = segment.get("text", "")
            if text.strip():
                embedding = embed_model.encode(text).tolist()
            else:
                embedding = [0.0] * 384 # embedding size of 384 for all-MiniLM-L6-v2

            enriched_segment = segment.copy()
            enriched_segment["embedding"] = embedding
            enriched_segments.append(enriched_segment)

        logger.info(f"Generated embeddings for {len(enriched_segments)} text segments")
        return enriched_segments
    except Exception:
        logger.exception("Failed to generate text embeddings")
        raise


def serialize_audio_embeddings(segments: List[Dict[str, Any]]) -> bytes:
    try:
        arrs = {}
        metadata = {}

        for seg in segments:
            seg_id = seg.get("id", 0)
            key = f"seg_{seg_id}"

            emb = np.array(seg.get("embedding", []), dtype=np.float32)
            arrs[key] = emb

            metadata[key] = {
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": str(seg.get("text", "")),
                "confidence": (float(seg.get("confidence")) if seg.get("confidence") is not None else None),
            }

        arrs["_metadata"] = np.array([json.dumps(metadata)], dtype=object)

        bio = io.BytesIO()
        np.savez_compressed(bio, **arrs)
        bio.seek(0)
        return bio.read()
    except Exception:
        logger.exception("Failed to serialize audio embeddings")
        raise


def deserialize_audio_embeddings(blob: Optional[bytes]) -> List[Dict[str, Any]]:
    if not blob:
        return []
    bio = io.BytesIO(blob)
    try:
        npz = np.load(bio, allow_pickle=True)
        segments = []

        metadata = {}
        if "_metadata" in npz.files:
            metadata = json.loads(str(npz["_metadata"][0]))

        for key in npz.files:
            if key.startswith("seg_"):
                seg_id = int(key.split("_")[1])
                embedding = npz[key].astype(np.float32).tolist()

                seg_meta = metadata.get(key, {})
                segment = {
                    "id": seg_id,
                    "start": seg_meta.get("start", 0.0),
                    "end": seg_meta.get("end", 0.0),
                    "text": seg_meta.get("text", ""),
                    "embedding": embedding,
                }
                segments.append(segment)

        segments.sort(key=lambda x: x["id"])
        return segments
    except Exception:
        logger.exception("Failed to deserialize audio embeddings")
        return []


def load_image_embedding_model():
    global _image_embedding_model
    if _image_embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _image_embedding_model = SentenceTransformer("clip-ViT-B-32")
        except Exception:
            logger.exception("Failed to load image embedding model; ensure sentence-transformers is installed and supports CLIP models")
            raise
    return _image_embedding_model


def generate_image_embedding(image: np.ndarray) -> List[float]:
    try:
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        model = load_image_embedding_model()
        emb = model.encode(pil)
        return emb.tolist()
    except Exception:
        logger.exception("Failed to generate image embedding")
        raise


def serialize_image_embedding(embedding: List[float], metadata: Dict[str, Any], key: str = "embedding") -> bytes:
    try:
        arrs = {}
        arrs[key] = np.array(embedding, dtype=np.float32)
        arrs["_metadata"] = np.array([json.dumps(metadata)], dtype=object)

        bio = io.BytesIO()
        np.savez_compressed(bio, **arrs)
        bio.seek(0)
        return bio.read()
    except Exception:
        logger.exception("Failed to serialize image embedding")
        raise


def deserialize_image_embedding(blob: Optional[bytes]) -> Dict[str, Any]:
    if not blob:
        return {}
    bio = io.BytesIO(blob)
    try:
        npz = np.load(bio, allow_pickle=True)
        result = {}
        if "_metadata" in npz.files:
            try:
                result["metadata"] = json.loads(str(npz["_metadata"][0]))
            except Exception:
                result["metadata"] = {}
        for key in npz.files:
            if key == "_metadata":
                continue
            result["embedding"] = npz[key].astype(np.float32).tolist()
            result["key"] = key
            break
        return result
    except Exception:
        logger.exception("Failed to deserialize image embedding blob")
        return {}

