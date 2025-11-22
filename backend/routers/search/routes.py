from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from models.database import get_db, Videos, Audios, VideoSummaries
from typing import List, Dict, Any
import json
import os

from routers.process.utils import load_embedding_model, deserialize_image_embedding, deserialize_audio_embeddings

router = APIRouter(prefix="", tags=["search"])


@router.get("/search")
async def search(
    q: str = Query(None),
    db_id: int = Query(None),
    similar_text: bool = Query(False),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    top_k: int = Query(10, ge=1, le=200),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    if db_id is not None:
        import numpy as np

        def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            if a is None or b is None:
                return 0.0
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 0.0
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        ref_row = db.query(Videos).filter(Videos.id == db_id).first()
        if not ref_row:
            raise HTTPException(status_code=404, detail="Video record not found")
        
        if not ref_row.embeddings:
            raise HTTPException(status_code=400, detail="Video record has no stored embedding")


        try:
            info = deserialize_image_embedding(ref_row.embeddings)
            query_emb = np.array(info.get("embedding", []), dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to load query embedding")

        results_sim = []
        rows = db.query(Videos).filter(Videos.embeddings != None).all()
        for r in rows:
            try:
                info = deserialize_image_embedding(r.embeddings)
                stored_emb = np.array(info.get("embedding", []), dtype=np.float32)
                score = _cosine_sim(query_emb, stored_emb)
                
                video_safe_name = os.path.splitext(r.filename or "")[0]
                kf_dir = f"{video_safe_name}_keyframes"
                metadata = info.get("metadata", {})
                
                if "frame_idx" in metadata and "timestamp" in metadata:
                    uploads_path = os.path.join(os.getcwd(), "uploads", kf_dir)
                    frame_idx = metadata["frame_idx"]
                    kf_filename = f"kf_{frame_idx}.jpg"
                    kf_path = os.path.join(uploads_path, kf_filename)
                    if os.path.exists(kf_path):
                        kf_url = f"/uploads/{kf_dir}/{kf_filename}"
                        metadata["url"] = kf_url
                    else:
                        keyframe_files = sorted([f for f in os.listdir(uploads_path) if f.endswith('.jpg')])
                        if keyframe_files:
                            kf_url = f"/uploads/{kf_dir}/{keyframe_files[0]}"
                            metadata["url"] = kf_url
                
                results_sim.append({"type": "video", "id": r.id, "filename": r.filename, "score": float(score), "metadata": metadata})
            except Exception:
                continue

        results_sim = sorted(results_sim, key=lambda x: x["score"], reverse=True)[:top_k]
        return {"results": results_sim, "pagination": {"page": 1, "per_page": top_k, "total_count": len(results_sim), "total_pages": 1}}

    if similar_text and q:
        import numpy as np
        try:
            model = load_embedding_model()
            q_emb = np.array(model.encode(q), dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to compute text embedding")

        def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            if a is None or b is None:
                return 0.0
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 0.0
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        results_sim = []
        rows = db.query(Audios).filter(Audios.embeddings != None).all()
        for r in rows:
            try:
                segments = deserialize_audio_embeddings(r.embeddings)
                for seg in segments:
                    emb = np.array(seg.get("embedding", []), dtype=np.float32)
                    if emb.size == 0:
                        continue
                    score = _cosine_sim(q_emb, emb)
                    results_sim.append({
                        "type": "audio",
                        "audio_id": r.id,
                        "filename": r.filename,
                        "segment_id": seg.get("id"),
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text"),
                        "score": float(score)
                    })
            except Exception:
                continue

        results_sim = sorted(results_sim, key=lambda x: x["score"], reverse=True)[:top_k]
        return {"results": results_sim, "pagination": {"page": 1, "per_page": top_k, "total_count": len(results_sim), "total_pages": 1}}

    if not q:
        raise HTTPException(status_code=400, detail="query param 'q' is required for text search")

    q_norm = q.strip().lower()
    if not q_norm:
        raise HTTPException(status_code=400, detail="query param 'q' must contain non-space characters")

    # Query videos by filename or detected objects
    video_candidates = db.query(Videos).filter(
        or_(
            Videos.filename.ilike(f"%{q}%"),
            Videos.detected_objects.ilike(f"%{q}%")
        )
    ).all()

    # Load per-video summaries into a dict for snippet assembly and matching
    try:
        summary_rows = db.query(VideoSummaries).all()
        summary_map = {s.filename: s.summary for s in summary_rows}
    except Exception:
        summary_map = {}

    audio_candidates = db.query(Audios).filter(
        or_(
            Audios.filename.ilike(f"%{q}%"),
            Audios.transcriptions.ilike(f"%{q}%")
        )
    ).all()

    results: List[Dict[str, Any]] = []

    def _score_and_snippet(text: str, field_name: str) -> Dict[str, Any]:
        txt = (text or "").lower()
        count = txt.count(q_norm)
        idx = txt.find(q_norm)
        snippet = None
        if idx != -1:
            start = max(0, idx - 40)
            end = min(len(txt), idx + len(q_norm) + 40)
            snippet = text[start:end]
        return {"count": count, "snippet": snippet, "field": field_name}

    for v in video_candidates:
        combined = " ".join([v.filename or "", v.detected_objects or "", summary_map.get(v.filename, "") or ""]) 
        info = _score_and_snippet(combined, "video_summary_or_filename")
        if info["count"] > 0:
            video_safe_name = os.path.splitext(v.filename or "")[0]
            kf_dir = f"{video_safe_name}_keyframes"
            keyframes = []
            
            uploads_path = os.path.join(os.getcwd(), "uploads", kf_dir)
            if os.path.exists(uploads_path):
                try:
                    keyframe_files = sorted([f for f in os.listdir(uploads_path) if f.endswith('.jpg')])
                    for i, kf_file in enumerate(keyframe_files):
                        url = f"/uploads/{kf_dir}/{kf_file}"
                        keyframes.append({
                            "frame_idx": i,
                            "url": url,
                            "timestamp": None,
                            "detected_objects": []
                        })
                except Exception:
                    pass
            
            try:
                detected_objects = json.loads(v.detected_objects) if v.detected_objects else []
                if not isinstance(detected_objects, list):
                    detected_objects = [detected_objects]
            except Exception:
                detected_objects = []
            
            results.append({
                "type": "video",
                "id": v.id,
                "filename": v.filename,
                "score": info["count"],
                "snippet": info["snippet"],
                "keyframes": keyframes,
                "detected_objects": detected_objects
            })

    for a in audio_candidates:
        combined = " ".join([a.filename or "", a.transcriptions or ""])
        info = _score_and_snippet(combined, "audio_transcription_or_filename")
        if info["count"] > 0:
            results.append({
                "type": "audio",
                "id": a.id,
                "filename": a.filename,
                "score": info["count"],
                "snippet": info["snippet"],
                "transcriptions": a.transcriptions
            })

    results = sorted(results, key=lambda r: r["score"], reverse=True)
    
    total_count = len(results)
    total_pages = (total_count + per_page - 1) // per_page
    
    if page > total_pages and total_pages > 0:
        page = total_pages
    elif page < 1:
        page = 1
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_results = results[start_idx:end_idx]
    
    return {
        "results": page_results,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": total_pages
        }
    }
