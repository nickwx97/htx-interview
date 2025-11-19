from fastapi import APIRouter

router = APIRouter(prefix="", tags=[""])


@router.get("/health")
async def health_check():
    return {"status": "ok"}
