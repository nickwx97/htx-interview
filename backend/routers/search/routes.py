from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from models.database import get_db, Videos, Audios
from typing import List, Dict, Any

router = APIRouter(prefix="", tags=["search"])


@router.get("/search")
async def search(q: str = Query(..., min_length=1), top_k: int = Query(10, ge=1), db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """
    Performs a case-insensitive full-text search on:
      - object labels
      - transcribed text
      - media filenames

    Returns a list of matching media entries ranked by a simple occurrence-based score.
    """
    if not q:
        raise HTTPException(status_code=400, detail="query param 'q' is required")

    q_norm = q.strip().lower()
    if not q_norm:
        raise HTTPException(status_code=400, detail="query param 'q' must contain non-space characters")

    # Search Videos where filename or detected_objects contain the query
    video_candidates = db.query(Videos).filter(
        or_(
            Videos.filename.ilike(f"%{q}%"),
            Videos.detected_objects.ilike(f"%{q}%")
        )
    ).all()

    # Search Audios where filename or transcriptions contain the query
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
        # create a short snippet around first occurrence
        idx = txt.find(q_norm)
        snippet = None
        if idx != -1:
            start = max(0, idx - 40)
            end = min(len(txt), idx + len(q_norm) + 40)
            snippet = text[start:end]
        return {"count": count, "snippet": snippet, "field": field_name}

    # Score video candidates
    for v in video_candidates:
        combined = " ".join([v.filename or "", v.detected_objects or ""]) 
        info = _score_and_snippet(combined, "video_summary_or_filename")
        if info["count"] > 0:
            results.append({"type": "video", "id": v.id, "filename": v.filename, "score": info["count"], "snippet": info["snippet"]})

    # Score audio candidates
    for a in audio_candidates:
        combined = " ".join([a.filename or "", a.transcriptions or ""])
        info = _score_and_snippet(combined, "audio_transcription_or_filename")
        if info["count"] > 0:
            results.append({"type": "audio", "id": a.id, "filename": a.filename, "score": info["count"], "snippet": info["snippet"]})

    # Sort by score (occurrence count) desc, then return top_k
    results = sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]
    return results


