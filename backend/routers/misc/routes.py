
from fastapi import APIRouter

router = APIRouter(prefix="", tags=[""])


@router.get("/health")
async def health_check():
    # Simple health check endpoint used by orchestration/monitoring
    return {"status": "ok"}
