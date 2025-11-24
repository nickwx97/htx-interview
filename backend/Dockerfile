FROM python:3.12-slim

WORKDIR /app

# Reduce image size and keep layer count low: install system deps together
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		python3-opencv \
		build-essential \
		libglib2.0-0 \
		libsm6 \
		libxext6 \
		libxrender1 \
		ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

# Set conservative thread limits for native libs to avoid CPU oversubscription
ENV OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 \
	TORCH_NUM_THREADS=1 \
	PYTHONUNBUFFERED=1 \
	UVICORN_WORKERS=1

# install torch cpu only first to skip gpu packages
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Use a shell form so the env var `UVICORN_WORKERS` can be expanded at runtime.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
