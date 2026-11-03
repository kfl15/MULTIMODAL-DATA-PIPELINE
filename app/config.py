"""One configuration shared by Docker, FastAPI, scripts, and notebooks."""
import os
from dataclasses import dataclass, field
from pathlib import Path

FORMATS = {
    "audio": {".wav", ".mp3", ".m4a"},
    "text": {".txt", ".md"},
    "pdf": {".pdf"},
    "excel": {".xlsx", ".csv"},
    "image": {".png", ".jpg", ".jpeg", ".tif", ".tiff"},
    "video": {".mp4", ".mkv", ".mov"},
    "docs": {".docx"},
}

@dataclass
class Settings:
    root: Path = field(default_factory=lambda: Path(os.getenv(
        "PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve())
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"))
    vision_model: str = field(default_factory=lambda: os.getenv("VISION_MODEL", "granite3.2-vision:2b"))
    frame_interval: float = field(default_factory=lambda: float(os.getenv("VIDEO_FRAME_INTERVAL", "10")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "2048")))

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        if self.device not in {"auto", "cpu"}:
            raise ValueError("DEVICE must be auto or cpu")
        if not 0.1 <= self.frame_interval <= 3600:
            raise ValueError("VIDEO_FRAME_INTERVAL must be between 0.1 and 3600 seconds")
        if self.max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be positive")
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_URL must be an HTTP URL")

    @property
    def models(self):
        return Path(os.getenv("MODEL_DIR", self.root / "models")).resolve()

    def initialize(self):
        for group in ("input", "output"):
            for category in FORMATS:
                (self.root / group / category).mkdir(parents=True, exist_ok=True)
        for group in ("models", "runtime", "reference_data"):
            (self.root / group).mkdir(parents=True, exist_ok=True)
