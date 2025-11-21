FROM python:3.12-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y python3-opencv
RUN apt-get update && apt-get install -y build-essential libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg && rm -rf /var/lib/apt/lists/*

# install torch cpu only first to skip gpu packages
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
