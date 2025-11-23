# HTX Interview

Lightweight prototype for media processing, retrieval and search. The project contains a FastAPI backend that processes uploaded videos and audio (keyframe extraction, object detection, transcription, embeddings) and a Vite + React frontend that interacts with the API.

**Contents**
- `backend/` — FastAPI app, processing pipeline, models and DB.
- `frontend/` — Vite + React UI.
- `samples/` — sample media used by tests.

**Quick summary**
- Backend: FastAPI app in `backend/main.py` with routers under `backend/routers/`.
- Processing: helpers in `backend/routers/process/utils.py` perform keyframe extraction, MobileNet-SSD detection, Whisper transcription, and embedding creation (`sentence-transformers`).
- Storage: SQLite DB defined in `backend/models/database.py`. Embeddings and binary blobs are stored in `LargeBinary` columns as compressed numpy archives.

**Key conventions**
- Keyframes are saved as `kf_<frame_idx>.jpg` under `uploads/<video>_keyframes/`.
- MobileNet model files expected at `backend/models/MobileNetSSD_deploy.prototxt` and `backend/models/mobilenet_iter_73000.caffemodel` (the code will attempt to download them if missing).

**Prerequisites**
- Docker & docker-compose (recommended) OR
- Python 3.12.12 (venv/conda), Node.js (v24 LTS) and system libs for audio/video processing (`ffmpeg`, `libsm6`, etc.).

## Running (recommended: Docker)

The repo includes a `docker-compose.yml` that builds and runs both backend and frontend. From the project root:

```bash
docker-compose up --build
```

This exposes:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`


## Local development (without Docker)

### Backend

1. Use the provided `backend/setup_env.sh` helper (recommended). The script installs required system packages and Python dependencies into a Conda environment and mirrors the Dockerfile setup.

```bash
cd backend
./setup_env.sh
```

After the script completes, activate the conda environment (defaults to `htx`):

```bash
conda activate htx
```

Optional: download the MobileNet-SSD model files used by the processing pipeline:

```bash
conda activate htx
python -c "from routers.process.utils import ensure_mobilenet_model; ensure_mobilenet_model()"
```

If you prefer not to use Conda, you can use a virtualenv as a fallback (manual):

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run the backend (from `backend/`):

```bash
uvicorn main:app --reload --port 8000
```

### Frontend

1. Install dependencies and run dev server:

```bash
cd frontend
npm install
npm run dev
```

2. The frontend expects the backend base URL in `VITE_API_BASE` (set by the dev server or environment). By default development servers expect the backend at `http://localhost:8000`.

## API (useful endpoints)

- POST `/process/video` — upload a multipart video file to process (keyframes, object detection, embeddings).
- POST `/process/audio` — upload audio file for transcription & embeddings.
- GET `/videos` — list processed videos.
- GET `/transcriptions` — list transcriptions.
- GET `/search?q=your+query` — text search. Add `&similar_text=true` for text similarity, or `?db_id=123` to run image-similarity by DB id.

Check `backend/routers/` for the full set of routes and query options.

## Local testing instructions

- Create a top-level logs directory (optional, used below):

```bash
mkdir -p ./logs/tests
```

### Backend (pytest)

1. Create and activate a Conda environment (recommended). This mirrors the Dockerfile Python version:

```bash
cd backend
# create the conda env (only required once)
conda create -n htx python=3.12 -y

# activate it for this shell
conda activate htx
```

2. Install system packages (if on Debian/Ubuntu) and Python dependencies. The repository includes a helper script `backend/setup_env.sh` which performs these steps and installs packages into the `htx` env using `conda run`:

```bash
# Run the helper (it will prompt/require sudo for apt installs)
./setup_env.sh
```

You can also run the steps manually and capture install logs:

```bash
# install system packages (requires sudo)
sudo apt-get update && sudo apt-get install -y python3-opencv build-essential libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg

# install Python deps into the active conda env
conda run -n htx pip install --upgrade pip setuptools wheel
conda run -n htx pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu || true
conda run -n htx pip install --no-cache-dir -r requirements.txt > ../logs/tests/backend-install.log 2>&1 || true
```

3. Run the full test suite and capture output (from `backend/`):

```bash
# ensure the env is active or use conda run
conda run -n htx pytest -q 2>&1 | tee ../logs/tests/backend-pytest.log
echo "${PIPESTATUS[0]:-0}" > ../logs/tests/backend_exit_code
```


Notes:
- Tests use sample media in `samples/sample-videos` and `samples/sample-audios`. (if missing, furnish from the interview package sent)

### Frontend (Vitest + React Testing Library)

1. Ensure Node.js (v24 LTS) and `npm` are installed.

2. Install dependencies and capture the install log:

```bash
cd frontend
npm install --legacy-peer-deps > ../logs/tests/frontend-npm-install.log 2>&1 || true
```

3. Run the Vitest suite and capture output:

```bash
npx vitest --run --reporter=dot 2>&1 | tee ../logs/tests/frontend-vitest.log
echo "${PIPESTATUS[0]:-0}" > ../logs/tests/frontend_exit_code
```


### Logs
All logs will be available under `./logs/tests/`.

### Troubleshooting & tips
- If pytest can't find `main`, ensure you're running from the `backend/` directory or add the backend folder to `PYTHONPATH`.
- Some ML/audio libs may emit deprecation warnings (e.g., `pkg_resources` or `aifc`) depending on your environment; these are warnings only.

## Models

- MobileNet SSD Caffe model files are in `backend/models/` (if missing, `process.utils.ensure_mobilenet_model()` will try to download them).

## Storage & uploads

- Uploaded media and generated keyframes are written to `backend/uploads/` and served statically by FastAPI at `/uploads` (see `backend/main.py`).
- Embeddings and binary blobs are serialized via `numpy.savez_compressed` and stored in the SQLite DB (`LargeBinary` fields). If you change serialization formats, update `process/utils.py` helpers accordingly.
