"""FastAPI serves the local page and a small, documented processing API."""
import json
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask
from .config import Settings, FORMATS
from .files import list_files, safe_path
from .jobs import Jobs


class JobRequest(BaseModel):
    categories: List[str]


def readiness(settings):
    required = ["whisper-tiny.en/model.bin", "layoutreader/config.json", "layoutreader/model.safetensors"]
    for model in ["PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec", "PP-DocLayout-S"]:
        required.extend(f"paddle/{model}/{file}" for file in ["inference.json", "inference.pdiparams", "inference.yml"])
    missing = [name for name in required if not (settings.models / name).is_file()]
    tools = {name: bool(shutil.which(name)) for name in ["ffmpeg", "ffprobe", "libreoffice"]}
    error = None
    available = False
    try:
        from .engines import Engines
        tags = Engines(settings).request("/api/tags")
        available = any(m["name"] == settings.vision_model for m in tags.get("models", []))
        if not available:
            error = f"Run ollama pull {settings.vision_model} on the host"
    except Exception as exc:
        error = str(exc)
    return {"ready": not missing and all(tools.values()) and available, "missing_models": missing,
            "tools": tools, "ollama": {"available": available, "error": error}, "device_policy": settings.device}


def create_app(settings=None, processor_factory=None):
    settings = settings or Settings()
    settings.initialize()
    jobs = Jobs(settings, **({"processor_factory": processor_factory} if processor_factory else {}))

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(title="Multimodal Preprocessor", lifespan=lifespan)
    app.state.jobs = jobs
    app.state.settings = settings
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "active_job": jobs.active, "formats": {k: sorted(v) for k, v in FORMATS.items()},
                "device_policy": settings.device}

    @app.get("/api/ready")
    def ready():
        return readiness(settings)

    @app.get("/api/files/{group}")
    def files(group: str):
        if group not in {"input", "output"}:
            raise HTTPException(400, "Choose input or output")
        return list_files(settings, group)

    @app.post("/api/upload/{category}")
    def upload(category: str, files: List[UploadFile] = File(...), paths: str = Form("[]")):
        if category not in FORMATS:
            raise HTTPException(400, "Unknown category")
        try:
            names = json.loads(paths)
            if not isinstance(names, list) or (names and len(names) != len(files)):
                raise ValueError("Upload path count does not match files")
            names = names or [file.filename for file in files]
            targets = []
            for name in names:
                if not isinstance(name, str):
                    raise ValueError("Upload paths must be strings")
                target = safe_path(settings.root / "input" / category, name)
                if target.suffix.lower() not in FORMATS[category]:
                    raise ValueError(f"Unsupported {category} file: {name}")
                targets.append(target)
            if len(set(targets)) != len(targets):
                raise ValueError("Duplicate upload paths")
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        # Do not replace an input while inference is reading it. Share the job lock.
        with jobs.lock:
            if jobs.active:
                raise HTTPException(409, "Wait for the active job before uploading")
            uploaded = []
            for file, target in zip(files, targets):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".upload-", delete=False) as handle:
                        temporary = Path(handle.name)
                        size = 0
                        while chunk := file.file.read(1024 * 1024):
                            size += len(chunk)
                            if size > settings.max_upload_mb * 1024**2:
                                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
                            handle.write(chunk)
                    temporary.replace(target)
                    uploaded.append(target.relative_to(settings.root / "input").as_posix())
                finally:
                    if temporary:
                        temporary.unlink(missing_ok=True)
                    file.file.close()
        return {"uploaded": uploaded}

    @app.post("/api/jobs", status_code=202)
    def start(request: JobRequest):
        try:
            return jobs.start(request.categories)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/jobs")
    def history():
        return jobs.list()

    def job_or_404(job_id):
        try:
            return jobs.get(job_id)
        except KeyError as error:
            raise HTTPException(404, "Job not found") from error

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return job_or_404(job_id)

    @app.get("/api/jobs/{job_id}/download")
    def bundle(job_id: str):
        report = job_or_404(job_id)
        if report["status"] in {"queued", "running"}:
            raise HTTPException(409, "Wait for the job to finish")
        with jobs.lock:
            if jobs.active:
                raise HTTPException(409, "Wait for the active job before packaging outputs")
            temporary = tempfile.NamedTemporaryFile(suffix=".zip", dir=settings.root / "runtime", delete=False)
            temporary.close()
            target = Path(temporary.name)
            try:
                with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                    for record in report["files"]:
                        if record.get("output"):
                            source = safe_path(jobs.directory / job_id / "outputs", record["output"])
                            if source.is_file():
                                archive.write(source, record["output"])
                    archive.writestr("job-report.json", json.dumps(report, indent=2))
            except Exception:
                target.unlink(missing_ok=True)
                raise
        return FileResponse(target, filename=f"text-outputs-{job_id[:8]}.zip",
                            background=BackgroundTask(target.unlink, missing_ok=True))

    def output_path(relative):
        try:
            target = safe_path(settings.root / "output", relative)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if not target.is_file() or target.suffix != ".txt":
            raise HTTPException(404, "Text output not found")
        return target

    @app.get("/api/preview")
    def preview(path: str):
        target = output_path(path)
        with target.open(encoding="utf-8") as handle:
            content = handle.read(100001)
        if len(content) > 100000:
            content = content[:100000] + "\n[Preview shortened. Download for the full text.]"
        return PlainTextResponse(content)

    @app.get("/api/download")
    def download(path: str):
        target = output_path(path)
        return FileResponse(target, filename=target.name, media_type="text/plain")

    return app


# Uvicorn uses the factory to avoid creating duplicate workers during imports.
# Command: uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
