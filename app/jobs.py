"""A single background worker, with durable reports and atomic text exports."""
import copy
import json
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from .files import list_files, safe_path, save_json, atomic_text
from .processors import Processor


def now():
    return datetime.now(timezone.utc).isoformat()


class Jobs:
    def __init__(self, settings, processor_factory=Processor):
        self.settings = settings
        self.factory = processor_factory
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.reports = {}
        self.active = None
        self.directory = settings.root / "runtime/jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("*.json"):
            try:
                report = json.loads(path.read_text())
                if report["status"] in {"queued", "running"}:
                    report.update(status="interrupted", stage="Interrupted by application restart", finished=now())
                    for record in report["files"]:
                        if record["status"] in {"queued", "running"}:
                            record["status"] = "interrupted"
                    save_json(path, report)
                self.reports[report["id"]] = report
            except (OSError, ValueError, KeyError):
                # One damaged report must not prevent access to the input/output folders.
                continue

    def persist(self, report):
        save_json(self.directory / (report["id"] + ".json"), report)

    def start(self, categories):
        from .config import FORMATS
        if not categories or any(c not in FORMATS for c in categories):
            raise ValueError("Select at least one valid category")
        with self.lock:
            if self.active:
                raise RuntimeError("Another job is running; wait for it to finish")
            files = [dict(item, status="queued") for item in list_files(self.settings, "input")
                     if item["category"] in categories]
            if not files:
                raise ValueError("No supported input files in the selected categories")
            report = {"id": uuid.uuid4().hex, "created": now(), "finished": None,
                      "status": "queued", "stage": "Queued", "completed": 0, "files": files}
            self.reports[report["id"]] = report
            self.active = report["id"]
            self.persist(report)
            self.executor.submit(self.run, report["id"])
            return copy.deepcopy(report)

    def progress(self, job_id, message):
        with self.lock:
            report = self.reports[job_id]
            report["stage"] = message
            self.persist(report)

    def run(self, job_id):
        report = self.reports[job_id]
        try:
            with self.lock:
                report["status"] = "running"
                self.persist(report)
            processor = self.factory(self.settings, progress=lambda msg: self.progress(job_id, msg))
            for item in report["files"]:
                with self.lock:
                    item["status"] = "running"
                    report["current_file"] = item["path"]
                    self.persist(report)
                try:
                    source = safe_path(self.settings.root / "input", item["path"])
                    result = processor.process(source, item["category"])
                    relative = item["path"] + ".txt"
                    destination = safe_path(self.settings.root / "output", relative)
                    atomic_text(destination, result.text(item["path"]))
                    # Preserve this job's version when a later rerun replaces current outputs.
                    snapshot = safe_path(self.directory / job_id / "outputs", relative)
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, snapshot)
                    with self.lock:
                        item.update(status="partial" if result.warnings else "completed", output=relative,
                                    warnings=result.warnings, devices=result.devices, details=result.details)
                except Exception as error:
                    with self.lock:
                        item.update(status="failed", error=str(error))
                with self.lock:
                    report["completed"] += 1
                    self.persist(report)
            with self.lock:
                states = {item["status"] for item in report["files"]}
                report["status"] = "completed" if states == {"completed"} else "failed" if states == {"failed"} else "partial"
                report["stage"] = "Finished"
        except Exception as error:
            with self.lock:
                report.update(status="failed", stage=str(error))
        finally:
            with self.lock:
                report["finished"] = now()
                self.active = None
                self.persist(report)

    def get(self, job_id):
        with self.lock:
            if job_id not in self.reports:
                raise KeyError(job_id)
            return copy.deepcopy(self.reports[job_id])

    def list(self):
        with self.lock:
            return [{k: report.get(k) for k in ("id", "created", "status", "stage", "completed")}
                    for report in sorted(self.reports.values(), key=lambda r: r["created"], reverse=True)]

    def close(self):
        self.executor.shutdown(wait=True)
