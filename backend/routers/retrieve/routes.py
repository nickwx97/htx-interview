from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.database import Videos, Audios, VideoSummaries, get_db
import os

router = APIRouter(prefix="", tags=["retrieval"])


def _video_keyframes_for_filename(filename: str):
    base = os.path.splitext(os.path.basename(filename))[0]
    uploads_dir = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
    kf_dir = os.path.join(uploads_dir, f"{base}_keyframes")
    out = []
    if os.path.isdir(kf_dir):
        try:
            for fname in sorted(os.listdir(kf_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    url = f"/uploads/{base}_keyframes/{fname}"

                    frame_idx = None
                    try:
                        name_part = os.path.splitext(fname)[0]
                        if name_part.startswith('kf_'):
                            frame_idx = int(name_part.split('kf_')[-1])
                    except Exception:
                        frame_idx = None
                    out.append({"filename": fname, "url": url, "frame_idx": frame_idx})
        except Exception:
            pass
    return out


@router.get("/videos")
async def retrieve_videos(db: Session = Depends(get_db)):
    rows = db.query(Videos).order_by(Videos.filename.asc(), Videos.frame_idx.asc(), Videos.created_at.asc()).all()
    groups = {}
    for r in rows:
        fn = r.filename or "unknown"
        if fn not in groups:
            groups[fn] = {"filename": fn, "rows": [], "keyframes": _video_keyframes_for_filename(fn)}
        groups[fn]["rows"].append({
            "id": r.id,
            "detected_objects": r.detected_objects,
            "frame_timestamps": r.frame_timestamps,
            "frame_idx": getattr(r, 'frame_idx', None),
            "embeddings": r.embeddings.hex() if r.embeddings else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    out = []
    for g in groups.values():
        # attach a video-level summary if available
        try:
            summary_row = db.query(VideoSummaries).filter(VideoSummaries.filename == g.get('filename')).order_by(VideoSummaries.created_at.desc()).first()
            if summary_row:
                g['summary'] = summary_row.summary
            else:
                g['summary'] = None
        except Exception:
            g['summary'] = None
        frame_idxs = set()
        for row in g.get('rows', []):
            if row.get('frame_idx') is not None:
                try:
                    frame_idxs.add(int(row.get('frame_idx')))
                except Exception:
                    pass

        filtered_kfs = []
        for kf in g.get('keyframes', []):
            kf_idx = kf.get('frame_idx')
            if kf_idx is not None:
                if kf_idx in frame_idxs:
                    filtered_kfs.append(kf)
            else:
                if len(g.get('rows', [])) > 0:
                    filtered_kfs.append(kf)

        try:
            filtered_kfs.sort(key=lambda x: (x.get('frame_idx') is None, x.get('frame_idx') if x.get('frame_idx') is not None else 0))
        except Exception:
            pass

        g['keyframes'] = filtered_kfs
        out.append(g)
    return out


@router.get("/transcriptions")
async def retrieve_audios(db: Session = Depends(get_db)):
    rows = db.query(Audios).order_by(Audios.created_at.desc()).all()
    groups = {}
    for r in rows:
        fn = r.filename or "unknown"
        if fn not in groups:
            groups[fn] = {"filename": fn, "rows": []}
        groups[fn]["rows"].append({
            "id": r.id,
            "transcriptions": r.transcriptions,
            "timestamps": r.timestamps,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return list(groups.values())
