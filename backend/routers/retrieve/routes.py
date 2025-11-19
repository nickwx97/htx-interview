from fastapi import APIRouter

router = APIRouter(prefix="", tags=["retrieval"])


@router.get("/videos")
async def retrieve_videos():
    pass


@router.get("/audios")
async def retrieve_audios():
    pass
