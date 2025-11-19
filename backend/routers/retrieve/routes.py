from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.database import Videos, get_db

router = APIRouter(prefix="", tags=["retrieval"])


@router.get("/videos")
async def retrieve_videos(db: Session = Depends(get_db)):
    videos = db.query(Videos).all()
    return videos


@router.get("/audios")
async def retrieve_audios(db: Session = Depends(get_db)):
    pass
