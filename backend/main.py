import uvicorn
from fastapi import FastAPI


app = FastAPI(
    title="Multimedia Processing & Analysis API",
    description="API for processing and analyzing video and audio content.",
    version="1.0",
)

from routers.misc.routes import router as api_router
from routers.process.routes import router as process_router
from routers.retrieve.routes import router as retrieve_router
from routers.search.routes import router as search_router

app.include_router(api_router)
app.include_router(process_router)
app.include_router(retrieve_router)
app.include_router(search_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
