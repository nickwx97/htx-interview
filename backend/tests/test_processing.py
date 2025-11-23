import os
import json
import cv2
import numpy as np
import tempfile
from pathlib import Path

import pytest

from routers.process.utils import (
    scene_keyframes,
    load_mobilenet_net,
    detect_objects_on_image,
    serialize_audio_embeddings,
    deserialize_audio_embeddings,
)


SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"


def test_scene_keyframes_and_detection_runs():
    """Run scene keyframe extraction on a sample video and run object detection on a keyframe.

    This verifies the frame extraction pipeline and that the detection helper returns the expected structure.
    """
    video_path = SAMPLES_DIR / "sample-videos" / "video_01.mp4"
    assert video_path.exists(), f"Sample video not found at {video_path}"

    with tempfile.TemporaryDirectory() as outdir:
        kfs = scene_keyframes(str(video_path), sample_rate=150, diff_thresh=1.0, output_dir=outdir)
        # should produce at least one keyframe with a file path
        assert isinstance(kfs, list)
        assert len(kfs) >= 1
        assert "frame_path" in kfs[0]

        # load the saved image and run detection
        img_path = kfs[0]["frame_path"]
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)
        net = load_mobilenet_net()
        annotated, objs = detect_objects_on_image(net, img, conf_thresh=0.2, draw_boxes=True)
        assert annotated is not None
        assert isinstance(objs, list)
        # each object should be a dict with label/confidence/bbox
        for o in objs:
            assert "label" in o and "confidence" in o and "bbox" in o


def test_serialize_deserialize_audio_embeddings_roundtrip():
    """Ensure audio embedding serialization/deserialization roundtrips metadata correctly."""
    segments = [
        {"id": 0, "start": 0.0, "end": 1.0, "text": "hello", "confidence": 0.9, "embedding": [0.1, 0.2, 0.3]},
        {"id": 1, "start": 1.0, "end": 2.0, "text": "world", "confidence": 0.8, "embedding": [0.4, 0.5, 0.6]},
    ]
    blob = serialize_audio_embeddings(segments)
    assert isinstance(blob, (bytes, bytearray))
    out = deserialize_audio_embeddings(blob)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["id"] == 0
    assert "embedding" in out[0]


def test_process_endpoints_and_search(monkeypatch, tmp_path, client):
    """Smoke test: POST /process/video and /process/audio return jobs (with scheduling stubbed),
    and search endpoint returns inserted rows.

    This test stubs background scheduling to avoid heavy processing.
    """
    from routers.process import routes as process_routes
    from models.database import SessionLocal, Videos, Audios

    # stub schedule_job to avoid background processing
    monkeypatch.setattr(process_routes, "schedule_job", lambda *a, **k: None)

    video_file = SAMPLES_DIR / "sample-videos" / "video_01.mp4"
    audio_file = SAMPLES_DIR / "sample-audios" / "Sample 1.mp3"

    # POST video
    with open(video_file, "rb") as vf:
        r = client.post("/process/video", files={"files": (video_file.name, vf, "video/mp4")})
    assert r.status_code == 200
    data = r.json()
    assert "jobs" in data and isinstance(data["jobs"], list)

    # POST audio
    with open(audio_file, "rb") as af:
        r2 = client.post("/process/audio", files={"files": (audio_file.name, af, "audio/mpeg")})
    assert r2.status_code == 200
    data2 = r2.json()
    assert "jobs" in data2 and isinstance(data2["jobs"], list)

    # Insert minimal video/audio rows and run a text search
    db = SessionLocal()
    try:
        v = Videos(filename="test_video.mp4", detected_objects=json.dumps([{"label": "person", "confidence": 0.9}]), frame_timestamps=json.dumps([0.0]), frame_idx=0)
        a = Audios(filename="test_audio.mp3", transcriptions="hello unit test", timestamps=json.dumps({"start": 0, "end": 1}))
        db.add(v)
        db.add(a)
        db.commit()
        db.refresh(v)
        db.refresh(a)

        # search for 'person' should return video
        rsearch = client.get("/search", params={"q": "person", "page": 1, "per_page": 10})
        assert rsearch.status_code == 200
        sr = rsearch.json()
        assert "results" in sr and isinstance(sr["results"], list)
        assert any(x.get("type") == "video" for x in sr["results"]) or any(x.get("type") == "audio" for x in sr["results"]) 
    finally:
        db.query(Videos).filter(Videos.filename == "test_video.mp4").delete()
        db.query(Audios).filter(Audios.filename == "test_audio.mp3").delete()
        db.commit()
        db.close()
