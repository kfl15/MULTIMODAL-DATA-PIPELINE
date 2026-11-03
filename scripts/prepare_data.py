"""Copy the owner's existing data without changing original notebook paths.

Run from the new project: python -m scripts.prepare_data --source-root ..
This script is optional and is not needed by people using their own inputs.
"""
import argparse
import hashlib
import shutil
from pathlib import Path
from app.config import Settings
from app.files import save_json


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    settings = Settings(root=destination)
    settings.initialize()
    mappings = []
    course = source / "data/course"
    for file in sorted((course / "amicorpus").glob("*/audio/*.wav")):
        mappings.append((file, Path("input/audio") / file.name))
    for file in sorted((course / "amicorpus").glob("*/slides/*.jpg")):
        mappings.append((file, Path("input/image/slides") / file.parent.parent.name / file.name))
    for lesson, name in [("L6", "output_7_1.png"), ("L9", "output_11_7.jpg"),
                          ("L9", "output_11_9.jpg"), ("L9", "output_27_0.png")]:
        file = source / lesson / name
        if file.is_file():
            mappings.append((file, Path("input/image/documents") / lesson / name))
    for file in sorted(course.rglob("*")):
        if file.is_file() and file.suffix in {".json", ".jsonl", ".txt"}:
            mappings.append((file, Path("reference_data") / file.relative_to(course)))
    if not mappings:
        raise FileNotFoundError(f"No existing course sources found under {source}")
    records = []
    for original, relative in mappings:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(original)
        if target.exists() and digest(target) != checksum:
            raise FileExistsError(f"Refusing to overwrite different local data: {target}")
        if not target.exists():
            shutil.copy2(original, target)
        if digest(target) != checksum:
            raise RuntimeError(f"Copy checksum failed: {target}")
        records.append({"source": original.relative_to(source).as_posix(),
                        "destination": relative.as_posix(), "sha256": checksum,
                        "bytes": target.stat().st_size})
    save_json(destination / "runtime/data_manifest.json", {"files": records,
        "notes": ["Source files were copied, not moved.",
                  "L9 bank statement contains pre-existing annotations.",
                  "JSONL exports are reference data, not raw video."]})
    print(f"Copied/verified {len(records)} files; manifest: runtime/data_manifest.json")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, Settings().root)
