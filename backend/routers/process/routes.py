from fastapi import APIRouter, Depends, File, UploadFile, Query, HTTPException
from sqlalchemy.orm import Session
from models.database import (
    get_db,
    Videos,
    Audios,
    VideoSummaries,
    ProcessingJobs,
    SessionLocal,
)
from typing import List
import logging
import os
import shutil
import asyncio
import cv2
import time
import json

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
    serialize_embeddings,
    deserialize_embeddings,
    deserialize_image_embedding,
)

# Router and runtime configuration
router = APIRouter(prefix="/process", tags=["process"])

# Upload dir
UPLOAD_DIR = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = logging.getLogger("process.routes")

# Job queue and worker pool
JOB_QUEUE: asyncio.Queue = asyncio.Queue()
WORKER_TASKS = []
DEFAULT_WORKER_COUNT = int(os.getenv("PROCESS_WORKER_COUNT", "2"))
MAX_RETRIES = int(os.getenv("PROCESS_MAX_RETRIES", "3"))


def schedule_job(job_id: int, file_path: str, filename: str, media_type: str):
    """Schedule job into the in-process queue (non-blocking)."""
    try:
        JOB_QUEUE.put_nowait((job_id, file_path, filename, media_type))
        logger.info(f"Scheduled job {job_id} for {filename} ({media_type}) into queue")
    except Exception:
        logger.exception(f"Failed to schedule job {job_id}")


def start_worker_pool(count: int = DEFAULT_WORKER_COUNT):
    """Start worker tasks to process jobs from JOB_QUEUE."""
    loop = asyncio.get_event_loop()
    for i in range(count):
        t = loop.create_task(_worker_loop(i))
        WORKER_TASKS.append(t)
    logger.info(f"Started {count} processing workers")


async def _worker_loop(worker_id: int):
    logger.info(f"Worker {worker_id} started")
    while True:
        job_id, file_path, filename, media_type = await JOB_QUEUE.get()
        logger.info(f"Worker {worker_id} picked job {job_id} ({filename})")
        attempt = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            # persist attempt count
            try:
                s = SessionLocal()
                job = s.query(ProcessingJobs).filter(ProcessingJobs.id == job_id).first()
                if job:
                    job.attempts = attempt
                    s.add(job)
                    s.commit()
                    s.refresh(job)
                s.close()
            except Exception:
                logger.exception(f"Failed to update attempt count for job {job_id}")

            try:
                _update_job(job_id, status="processing", progress=5)
                if media_type == "video":
                    await _process_video_background(job_id, file_path, filename)
                else:
                    await _process_audio_background(job_id, file_path, filename)
                # success
                break
            except Exception as e:
                logger.exception(f"Job {job_id} failed on attempt {attempt}: {e}")
                if attempt >= MAX_RETRIES:
                    _update_job(job_id, status="error", progress=100, error=str(e))
                    logger.error(f"Job {job_id} exhausted retries and marked error")
                else:
                    _update_job(job_id, status="retrying", progress=0, error=str(e))
                    backoff = min(5 * attempt, 30)
                    logger.info(f"Retrying job {job_id} after {backoff}s (attempt {attempt}/{MAX_RETRIES})")
                    await asyncio.sleep(backoff)
        JOB_QUEUE.task_done()


@router.post("/video")
async def process_video(
    files: List[UploadFile] = File(...), db: Session = Depends(get_db)
):
    """Accept multiple video files, create a processing job per file, and run processing in background tasks.
    Returns list of job ids which can be polled via `/process/status/{job_id}`.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    jobs = []
    for f in files:
        filename = f.filename
        if not filename:
            continue
        # save uploaded file
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        # create job record
        job = _create_job(db, filename, "video")
        jobs.append({"job_id": job.id, "filename": filename, "status": job.status})

        # schedule background processing via central queue
        schedule_job(job.id, file_path, filename, "video")

    return {"jobs": jobs}


@router.post("/audio")
async def process_audio(
    files: List[UploadFile] = File(...), db: Session = Depends(get_db)
):
    """Accept multiple audio files, create a processing job per file, and run processing in background tasks.
    Returns list of job ids which can be polled via `/process/status/{job_id}`.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    jobs = []
    for f in files:
        filename = f.filename
        if not filename:
            continue
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        job = _create_job(db, filename, "audio")
        jobs.append({"job_id": job.id, "filename": filename, "status": job.status})

        schedule_job(job.id, file_path, filename, "audio")

    return {"jobs": jobs}


@router.get("/jobs")
async def list_jobs(
    page: int = Query(1, ge=1),
    per_page: int = Query(5, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Return paginated jobs (latest first).

    Query params:
    - page: 1-based page number
    - per_page: items per page (default 5)
    """
    total = db.query(ProcessingJobs).count()
    offset = (page - 1) * per_page
    rows = (
        db.query(ProcessingJobs)
        .order_by(ProcessingJobs.created_at.desc())
        .offset(offset)
        .limit(per_page)
        .all()
    )
    out = []
    for job in rows:
        try:
            res = json.loads(job.result) if job.result else None
        except Exception:
            res = job.result
        out.append(
            {
                "id": job.id,
                "filename": job.filename,
                "media_type": job.media_type,
                "status": job.status,
                "progress": job.progress,
                "result": res,
                "error": job.error,
                "created_at": job.created_at.isoformat() if job.created_at else None,
            }
        )
    return {"jobs": out, "page": page, "per_page": per_page, "total": total}


@router.get("/status/{job_id}")
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ProcessingJobs).filter(ProcessingJobs.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = None
    try:
        if job.result:
            result = json.loads(job.result)
    except Exception:
        result = job.result
    return {
        "id": job.id,
        "filename": job.filename,
        "media_type": job.media_type,
        "status": job.status,
        "progress": job.progress,
        "result": result,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


def _create_job(db: Session, filename: str, media_type: str):
    job = ProcessingJobs(
        filename=filename, media_type=media_type, status="queued", progress=0
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _update_job(
    job_id: int,
    status: str = None,
    progress: int = None,
    result: object = None,
    error: str = None,
):
    session = SessionLocal()
    try:
        job = session.query(ProcessingJobs).filter(ProcessingJobs.id == job_id).first()
        if not job:
            return None
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = int(progress)
        if result is not None:
            try:
                job.result = json.dumps(result)
            except Exception:
                job.result = str(result)
        if error is not None:
            job.error = str(error)
        session.add(job)
        session.commit()
        session.refresh(job)
        return job
    except Exception:
        logger.exception("Failed to update job")
        try:
            session.rollback()
        except Exception:
            pass
    finally:
        session.close()


async def _process_video_background(job_id: int, file_path: str, filename: str):
    def sync_video_job():
        logger.info(f"[Job {job_id}] Starting video processing: {filename}")
        _update_job(job_id, status="processing", progress=5)
        db = SessionLocal()
        try:
            video_safe_name = os.path.splitext(os.path.basename(filename))[0]
            kf_dir = os.path.join(UPLOAD_DIR, f"{video_safe_name}_keyframes")
            os.makedirs(kf_dir, exist_ok=True)

            try:
                sample_rate = int(os.getenv("VIDEO_SAMPLE_RATE", "5"))
            except Exception:
                sample_rate = 5
            try:
                diff_thresh = float(os.getenv("VIDEO_DIFF_THRESH", "30.0"))
            except Exception:
                diff_thresh = 30.0

            _update_job(job_id, progress=10)
            logger.info(f"[Job {job_id}] Keyframe extraction started for {filename}")
            keyframes_meta = scene_keyframes(
                file_path,
                sample_rate=sample_rate,
                diff_thresh=diff_thresh,
                output_dir=kf_dir,
            )
            logger.info(
                f"[Job {job_id}] Keyframe extraction complete: {len(keyframes_meta)} keyframes"
            )
            _update_job(job_id, progress=30)
            logger.info(f"[Job {job_id}] Object detection started for {filename}")

            net = load_mobilenet_net()
            detections_by_frame = []
            saved_keyframes = []

            for idx, kf_meta in enumerate(keyframes_meta):
                frame_idx = kf_meta.get("frame_idx")
                timestamp = kf_meta.get("timestamp")
                frame_path = kf_meta.get("frame_path")
                img = None
                if frame_path and os.path.exists(frame_path):
                    img = cv2.imread(frame_path)

                if img is None:
                    logger.warning(f"Failed to load keyframe from {frame_path}")
                    continue

                annotated_img, objs = detect_objects_on_image(
                    net, img, conf_thresh=0.7, draw_boxes=True
                )
                try:
                    cv2.imwrite(frame_path, annotated_img)
                except Exception:
                    logger.exception("Failed to write annotated keyframe")

                detections_by_frame.append(
                    {"frame_idx": frame_idx, "timestamp": timestamp, "objects": objs}
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

                if idx % 5 == 0:
                    prog = min(60, 30 + int((idx / max(1, len(keyframes_meta))) * 40))
                    _update_job(job_id, progress=prog)
                    logger.info(
                        f"[Job {job_id}] Processed {idx+1}/{len(keyframes_meta)} keyframes (progress: {prog}%)"
                    )

            summary = summarize_detections(detections_by_frame)
            logger.info(f"[Job {job_id}] Object detection complete for {filename}")
            _update_job(job_id, progress=65)
            logger.info(
                f"[Job {job_id}] Generating summary and embeddings for {filename}"
            )

            # compute embeddings and persist rows
            created_rows = []
            kf_map = {kf.get("frame_idx"): kf for kf in saved_keyframes}

            for i, frame in enumerate(detections_by_frame):
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
                    except Exception:
                        logger.exception(
                            "Failed to compute image embedding for keyframe"
                        )

                for obj in frame.get("objects", []):
                    try:
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
                    except Exception:
                        logger.exception("Failed to add video row")

                if i % 5 == 0:
                    try:
                        db.commit()
                        for r in created_rows:
                            try:
                                db.refresh(r)
                            except Exception:
                                pass
                    except Exception:
                        db.rollback()
                        logger.exception("Commit failed while inserting video rows")
                    _update_job(job_id, progress=75)
                    logger.info(
                        f"[Job {job_id}] Inserted {i+1}/{len(detections_by_frame)} detection frames (progress: 75%)"
                    )

            try:
                db.commit()
            except Exception:
                db.rollback()
                logger.exception("Final commit failed for video rows")

            # persist summary
            try:
                summary_text = summary
                if summary_text:
                    if not isinstance(summary_text, str):
                        summary_text = json.dumps(summary_text)
                    db.query(VideoSummaries).filter(
                        VideoSummaries.filename == filename
                    ).delete()
                    summary_row = VideoSummaries(
                        filename=filename, summary=summary_text
                    )
                    db.add(summary_row)
                    db.commit()
                    db.refresh(summary_row)
            except Exception:
                logger.exception("Failed to persist video summary")

            db_ids = [r.id for r in created_rows if getattr(r, "id", None) is not None]
            result = {
                "db_ids": db_ids,
                "keyframes": saved_keyframes,
                "summary": summary,
            }
            _update_job(job_id, status="done", progress=100, result=result)
            logger.info(f"[Job {job_id}] Video processing complete for {filename}")

        except Exception as e:
            logger.exception("Background video processing failed")
            _update_job(job_id, status="error", progress=100, error=str(e))
        finally:
            try:
                os.remove(file_path)
            except Exception:
                pass
            db.close()

    await asyncio.to_thread(sync_video_job)


async def _process_audio_background(job_id: int, file_path: str, filename: str):
    def sync_audio_job():
        logger.info(f"[Job {job_id}] Starting audio processing: {filename}")
        _update_job(job_id, status="processing", progress=5)
        db = SessionLocal()
        try:
            # transcribe
            logger.info(f"[Job {job_id}] Transcription started for {filename}")
            transcription_result = transcribe_audio(
                file_path, model_name="openai/whisper-tiny"
            )
            logger.info(f"[Job {job_id}] Transcription complete for {filename}")
            _update_job(job_id, progress=30)
            full_text = transcription_result.get("text", "")
            segments = transcription_result.get("segments", [])

            logger.info(f"[Job {job_id}] Generating text embeddings for {filename}")
            enriched_segments = generate_text_embeddings(segments)
            _update_job(job_id, progress=60)
            logger.info(f"[Job {job_id}] Embeddings generated for {filename}")

            created_rows = []
            for seg in enriched_segments:
                seg_id = seg.get("id", 0)
                seg_text = seg.get("text", "")
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                seg_embedding = seg.get("embedding", [])
                seg_conf = seg.get("confidence", None)

                embedding_blob = None
                if seg_embedding:
                    embedding_blob = serialize_audio_embeddings([seg])

                timestamps_json = json.dumps(
                    {
                        "id": seg_id,
                        "start": seg_start,
                        "end": seg_end,
                        "confidence": (
                            float(seg_conf) if seg_conf is not None else None
                        ),
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

            try:
                db.commit()
                for r in created_rows:
                    try:
                        db.refresh(r)
                    except Exception:
                        pass
            except Exception:
                db.rollback()
                logger.exception("Failed to commit audio rows")

            response_segments = [
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "confidence": seg.get("confidence", None),
                }
                for seg in enriched_segments
            ]

            result = {
                "filename": filename,
                "text": full_text,
                "segments": response_segments,
                "segment_count": len(response_segments),
                "db_ids": [
                    r.id for r in created_rows if getattr(r, "id", None) is not None
                ],
            }
            _update_job(job_id, status="done", progress=100, result=result)
            logger.info(f"[Job {job_id}] Audio processing complete for {filename}")

        except Exception as e:
            logger.exception("Background audio processing failed")
            _update_job(job_id, status="error", progress=100, error=str(e))
        finally:
            try:
                os.remove(file_path)
            except Exception:
                pass
            db.close()

    await asyncio.to_thread(sync_audio_job)
