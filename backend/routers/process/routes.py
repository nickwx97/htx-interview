from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session
from models.database import get_db, Videos, Audios, VideoSummaries
import logging
import os
import shutil
import asyncio
import cv2

from .utils import (
    scene_keyframes,
    load_mobilenet_net,
    detect_objects_on_image,
    summarize_detections,
    transcribe_audio,
    generate_text_embeddings,
    serialize_audio_embeddings,
    generate_image_embedding,
    serialize_image_embedding,
)

from fastapi import HTTPException
import json

router = APIRouter(prefix="/process", tags=["process"])

logger = logging.getLogger("API")

UPLOAD_DIR = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/video")
async def process_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def _process():
        try:
            video_safe_name = os.path.splitext(os.path.basename(filename))[0]
            kf_dir = os.path.join(UPLOAD_DIR, f"{video_safe_name}_keyframes")
            os.makedirs(kf_dir, exist_ok=True)

            # Allow sample rate and diff threshold to be configured via environment variables (backend/.env)
            try:
                sample_rate = int(os.getenv("VIDEO_SAMPLE_RATE", "5"))
            except Exception:
                sample_rate = 5
            try:
                diff_thresh = float(os.getenv("VIDEO_DIFF_THRESH", "30.0"))
            except Exception:
                diff_thresh = 30.0

            keyframes_meta = scene_keyframes(
                file_path, sample_rate=sample_rate, diff_thresh=diff_thresh, output_dir=kf_dir
            )
            net = load_mobilenet_net()
            detections_by_frame = []
            saved_keyframes = []

            batch_size = 10  # Process 10 frames at a time
            for batch_idx in range(0, len(keyframes_meta), batch_size):
                batch = keyframes_meta[batch_idx : batch_idx + batch_size]

                for kf_meta in batch:
                    frame_idx = kf_meta["frame_idx"]
                    timestamp = kf_meta["timestamp"]
                    frame_path = kf_meta["frame_path"]

                    img = cv2.imread(frame_path)
                    if img is None:
                        logger.warning(f"Failed to load keyframe from {frame_path}")
                        continue

                    annotated_img, objs = detect_objects_on_image(
                        net, img, conf_thresh=0.7, draw_boxes=True
                    )

                    cv2.imwrite(frame_path, annotated_img)

                    detections_by_frame.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                            "objects": objs,
                        }
                    )

                    rel_path = os.path.relpath(frame_path, os.getcwd())
                    url_path = f"/uploads/{rel_path.replace(os.path.sep, '/')}"
                    saved_keyframes.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                            "image_path": frame_path,
                            "url": url_path,
                        }
                    )

                # Force garbage collection after each batch to free memory
                import gc

                gc.collect()

            summary = summarize_detections(detections_by_frame)

            return {
                "keyframes": saved_keyframes,
                "detections": detections_by_frame,
                "summary": summary,
            }
        except Exception:
            logger.exception("Processing failed")
            raise

    result = await asyncio.to_thread(_process)

    try:
        created_rows = []
        summary_text = result.get("summary")
        kf_map = {kf["frame_idx"]: kf for kf in result.get("keyframes", [])}

        for frame in result.get("detections", []):
            frame_idx = frame.get("frame_idx")
            timestamp = frame.get("timestamp")

            embedding_blob = None
            kf = kf_map.get(frame_idx)
            if kf is not None:
                kf_path = kf.get("image_path")
                try:
                    if kf_path and os.path.exists(kf_path):
                        img = cv2.imread(kf_path)
                        if img is not None:
                            emb = generate_image_embedding(img)
                            embedding_blob = serialize_image_embedding(
                                emb,
                                {
                                    "frame_idx": frame_idx,
                                    "timestamp": timestamp,
                                    "filename": filename,
                                },
                            )
                        else:
                            logger.warning(
                                f"Could not read keyframe image for embedding: {kf_path}"
                            )
                    else:
                        logger.warning(
                            f"Keyframe path missing or not found for frame {frame_idx}"
                        )
                except Exception:
                    logger.exception(
                        f"Failed to compute image embedding for keyframe {kf_path}"
                    )

                for obj in frame.get("objects", []):
                    detected_object_json = json.dumps(obj)
                    frame_timestamp_json = json.dumps(timestamp)

                    video_row = Videos(
                        filename=filename,
                        detected_objects=detected_object_json,
                        frame_timestamps=frame_timestamp_json,
                        frame_idx=frame_idx,
                        embeddings=embedding_blob,
                    )
                    db.add(video_row)
                    created_rows.append(video_row)

        db.commit()
        for r in created_rows:
            db.refresh(r)
        result["db_ids"] = [r.id for r in created_rows]

        # Persist a single summary row per video filename. Replace any existing summary for the filename.
        try:
            if summary_text:
                if not isinstance(summary_text, str):
                    summary_text = json.dumps(summary_text)
                # remove existing summaries for this filename
                db.query(VideoSummaries).filter(VideoSummaries.filename == filename).delete()
                summary_row = VideoSummaries(filename=filename, summary=summary_text)
                db.add(summary_row)
                db.commit()
                db.refresh(summary_row)
                result["summary_id"] = summary_row.id
        except Exception:
            logger.exception("Failed to persist video summary")
    except Exception:
        logger.exception("Failed to insert video processing result(s) into DB")
        try:
            db.rollback()
        except Exception:
            logger.exception("Rollback failed")

    try:
        os.remove(file_path)
    except Exception:
        logger.warning(f"Could not remove uploaded video file: {file_path}")

    return result


@router.post("/audio")
async def process_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def _process():
        try:
            transcription_result = transcribe_audio(
                file_path, model_name="openai/whisper-tiny"
            )
            full_text = transcription_result.get("text", "")
            segments = transcription_result.get("segments", [])

            enriched_segments = generate_text_embeddings(segments)

            logger.info(
                f"Audio processing complete: {len(enriched_segments)} segments, text length={len(full_text)}"
            )

            return {
                "text": full_text,
                "segments": enriched_segments,
            }
        except Exception:
            logger.exception("Audio processing failed")
            raise

    result = await asyncio.to_thread(_process)

    try:
        full_text = result.get("text", "")
        segments = result.get("segments", [])
        created_rows = []

        for seg in segments:
            seg_id = seg.get("id", 0)
            seg_text = seg.get("text", "")
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg_confidence = seg.get("confidence", None)
            seg_embedding = seg.get("embedding", [])

            segment_json = json.dumps(
                {
                    "id": seg_id,
                    "text": seg_text,
                    "start": seg_start,
                    "end": seg_end,
                }
            )

            embedding_blob = None
            if seg_embedding:
                embedding_blob = serialize_audio_embeddings([seg])

            timestamps_json = json.dumps(
                {
                    "id": seg_id,
                    "start": seg_start,
                    "end": seg_end,
                    "confidence": (float(seg_confidence) if seg_confidence is not None else None),
                }
            )

            audio_row = Audios(
                filename=filename,
                transcriptions=seg_text,
                timestamps=timestamps_json,
                embeddings=embedding_blob,
            )
            db.add(audio_row)
            created_rows.append(audio_row)

        db.commit()
        for r in created_rows:
            db.refresh(r)

        response_segments = [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "confidence": seg.get("confidence", None),
            }
            for seg in segments
        ]

        try:
            os.remove(file_path)
        except Exception:
            logger.warning(f"Could not remove uploaded audio file: {file_path}")

        return {
            "filename": filename,
            "text": full_text,
            "segments": response_segments,
            "segment_count": len(segments),
            "db_ids": [r.id for r in created_rows],
        }
    except Exception:
        logger.exception("Failed to insert audio processing result(s) into DB")
        try:
            db.rollback()
        except Exception:
            logger.exception("Rollback failed")
        raise HTTPException(status_code=500, detail="DB insert failed")
