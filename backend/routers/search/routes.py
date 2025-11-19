from fastapi import APIRouter

router = APIRouter(prefix="", tags=["search"])


@router.get("/search")
async def search():
    pass
