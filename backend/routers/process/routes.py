from fastapi import APIRouter

router = APIRouter(prefix="/process", tags=["process"])


@router.post("/video")
async def process_video():
    pass


@router.post("/audio")
async def process_audio():
    pass
