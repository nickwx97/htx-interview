import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from backend/.env if present
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=env_path, override=False)
except Exception:
    # dotenv not available or failed to load — fall back to OS env
    pass


# FastAPI application entrypoint
app = FastAPI(
    title="Multimedia Processing & Analysis API",
    description="API for processing and analyzing video and audio content.",
    version="1.0",
)

# Configure CORS from environment variable `ALLOWED_ORIGINS` (comma-separated)
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_raw and allowed_origins_raw.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
from routers.misc.routes import router as api_router
from routers.process.routes import router as process_router
from routers.retrieve.routes import router as retrieve_router
from routers.search.routes import router as search_router

app.include_router(api_router)
app.include_router(process_router)
app.include_router(retrieve_router)
app.include_router(search_router)

# Serve uploaded media and generated assets from `uploads/`
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# Create database tables
from models.database import create_tables, SessionLocal, ProcessingJobs
create_tables()

# Resume unfinished jobs on startup (enqueue queued/processing jobs)
import asyncio
from routers.process.routes import schedule_job, start_worker_pool, UPLOAD_DIR
import os
from contextlib import asynccontextmanager


# Asynchronous job resumption on startup
async def resume_unfinished_jobs():
    loop = asyncio.get_event_loop()
    session = SessionLocal()
    try:
        jobs = (
            session.query(ProcessingJobs)
            .filter(ProcessingJobs.status.in_(["queued", "processing"]))
            .order_by(ProcessingJobs.created_at.asc())
            .all()
        )
        for job in jobs:
            file_path = os.path.join(UPLOAD_DIR, job.filename)
            if not os.path.exists(file_path):
                # Can't resume if file is missing, mark as failed
                job.status = "failed"
                job.error = "Source file missing; cannot resume processing."
                session.commit()
                continue  # Can't resume if file is missing
            # schedule into queue for workers
            schedule_job(job.id, file_path, job.filename, job.media_type)
        await asyncio.sleep(0)  # Yield control to event loop
    finally:
        session.close()

@asynccontextmanager
async def lifespan(app):
    # Lifespan handler: start worker pool, then enqueue previously queued jobs
    start_worker_pool()
    await resume_unfinished_jobs()
    try:
        yield
    finally:
        # shutdown/cleanup can be added here if needed
        pass

# Attach the lifespan context to the app (compatible with FastAPI/Starlette)
app.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
