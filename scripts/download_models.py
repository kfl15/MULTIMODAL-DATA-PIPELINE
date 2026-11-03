"""Download pinned local weights; record checksums and the host Ollama digest."""
import argparse
import hashlib
import json
from pathlib import Path
from app.config import Settings
from app.engines import Engines
from app.files import save_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(verify_only=False):
    settings = Settings()
    settings.initialize()
    lock = json.loads((Path(__file__).resolve().parents[1] / "models.lock.json").read_text())
    report = {}
    for local, spec in lock.items():
        directory = settings.models / local
        missing = [name for name in spec["checksums"] if not (directory / name).is_file()]
        if missing:
            if verify_only:
                raise FileNotFoundError(f"Missing {local}: {missing}")
            from huggingface_hub import snapshot_download
            snapshot_download(spec["repo_id"], revision=spec["revision"], local_dir=directory,
                              allow_patterns=spec["patterns"], max_workers=2)
        files = {}
        for name, expected in spec["checksums"].items():
            path = directory / name
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"Checksum mismatch: {path}. Remove this damaged file and rerun downloads.")
            files[name] = {"sha256": actual, "bytes": path.stat().st_size}
        # Retain the LayoutReader model card even when weights were already present.
        if local == "layoutreader" and not (directory / "README.md").exists() and not verify_only:
            from huggingface_hub import hf_hub_download
            hf_hub_download(spec["repo_id"], "README.md", revision=spec["revision"], local_dir=directory)
        report[local] = {"repo_id": spec["repo_id"], "revision": spec["revision"], "files": files}
        print(f"Verified {local}", flush=True)
    tags = Engines(settings).request("/api/tags")
    model = next((m for m in tags.get("models", []) if m["name"] == settings.vision_model), None)
    if model is None:
        raise RuntimeError(f"On the host, run: ollama pull {settings.vision_model}")
    report["ollama"] = {"model": settings.vision_model, "digest": model["digest"]}
    save_json(settings.root / "runtime/model_manifest.json", report)
    print("Models ready; identities saved in runtime/model_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    main(parser.parse_args().verify_only)
