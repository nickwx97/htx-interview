from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session
from models.database import get_db, Videos, Audios
import logging
import os
import shutil
import asyncio
from typing import List, Dict, Any

from .utils import (
    scene_keyframes,
    load_mobilenet_net,
    detect_objects_on_image,
    summarize_detections,
)
from .utils import serialize_embeddings

from fastapi import HTTPException
import json

router = APIRouter(prefix="/process", tags=["process"])

logger = logging.getLogger("API")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/video")
async def process_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Accepts an uploaded video file, runs scene detection + object detection on keyframes,
    computes embeddings for detected labels, and returns a summary.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    file_path = os.path.join(UPLOAD_DIR, filename)
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run processing in thread to avoid blocking event loop
    def _process():
        try:
            keyframes = scene_keyframes(file_path, sample_rate=5, diff_thresh=25.0)
            net = load_mobilenet_net()
            detections_by_frame = []
            for kf in keyframes:
                img = kf["frame"]
                objs = detect_objects_on_image(net, img, conf_thresh=0.45)
                detections_by_frame.append({"frame_idx": kf["frame_idx"], "timestamp": kf["timestamp"], "objects": objs})
            summary = summarize_detections(detections_by_frame)
            return {"keyframes": [{"frame_idx": kf["frame_idx"], "timestamp": kf["timestamp"]} for kf in keyframes], "detections": detections_by_frame, "summary": summary}
        except Exception:
            logger.exception("Processing failed")
            raise

    result = await asyncio.to_thread(_process)

    # Insert results into DB
    try:
        detected_objects_json = json.dumps(result.get("detections", []))
        frame_timestamps_json = json.dumps([kf["timestamp"] for kf in result.get("keyframes", [])])
        # serialize embeddings from summary into a binary blob
        embeddings_blob = serialize_embeddings(result.get("summary", []))
        video_row = Videos(filename=filename, detected_objects=detected_objects_json, frame_timestamps=frame_timestamps_json, embeddings=embeddings_blob)
        db.add(video_row)
        db.commit()
        db.refresh(video_row)
        # include db id in response
        result["db_id"] = video_row.id
    except Exception:
        logger.exception("Failed to insert video processing result into DB")

    return result

@router.post("/audio")
async def process_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    pass