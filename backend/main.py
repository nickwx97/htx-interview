import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

# Load environment variables from backend/.env if present
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=env_path, override=False)
except Exception:
    # dotenv not available or failed to load — fall back to OS env
    pass


app = FastAPI(
    title="Multimedia Processing & Analysis API",
    description="API for processing and analyzing video and audio content.",
    version="1.0",
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

# Serve uploaded media and generated assets
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Create database tables
from models.database import create_tables
create_tables()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
