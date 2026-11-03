"""Exercise real model inference; save failures as well as successful device reports.

Uses short existing audio/image samples if present, otherwise explicitly labeled
synthetic validation fixtures. Does not claim GPU execution from device discovery.
"""
import argparse
import gc
import json
import tempfile
import wave
from pathlib import Path
from app.config import Settings
from app.engines import Engines
from app.files import save_json


def main(require_gpu=False):
    settings = Settings(); settings.initialize()
    engines = Engines(settings)
    report = {"device_policy": settings.device, "checks": {}, "events": engines.events}
    def check(name, callback):
        try:
            detail = callback()
            report["checks"][name] = {"passed": True, "detail": detail}
        except Exception as error:
            report["checks"][name] = {"passed": False, "error": str(error)}
        finally:
            save_json(settings.root / "runtime/runtime_verification.json", report)
    with tempfile.TemporaryDirectory(dir=settings.root / "runtime", prefix="verification-") as directory:
        work = Path(directory)
        image_path = next((settings.root / "input/image/documents/L6").glob("*.png"), None)
        if image_path is None:
            image_path = next((settings.root / "input/image").rglob("*.jpg"), None)
        if image_path is None:
            from PIL import Image, ImageDraw
            image_path = work / "synthetic-validation.png"
            image = Image.new("RGB", (640, 200), "white")
            ImageDraw.Draw(image).text((30, 50), "Local preprocessing validation", fill="black")
            image.save(image_path)
            report["image_source"] = "Synthetic validation fixture; not owner data"
        else:
            report["image_source"] = str(image_path.relative_to(settings.root))
        engines.unload_vision()
        def torch_probe():
            import torch
            device = "cuda" if settings.device != "cpu" and torch.cuda.is_available() else "cpu"
            tensor = torch.ones((4, 4), device=device) @ torch.ones((4, 4), device=device)
            assert tensor[0, 0].item() == 4
            if require_gpu: assert tensor.device.type == "cuda", "Torch GPU required"
            engines.record("Torch computation", str(tensor.device), verified=True)
            return {"torch": torch.__version__, "cuda_build": torch.version.cuda, "device": str(tensor.device)}
        check("torch", torch_probe)
        check("ocr", lambda: {"regions": len(engines.ocr(image_path))})
        check("layout", lambda: {"regions": len(engines.layout(image_path))})
        check("reading_order", lambda: engines.order([
            {"bbox": [10, 10, 200, 40]}, {"bbox": [10, 70, 200, 100]}], 640, 200))
        def whisper():
            source = next((settings.root / "input/audio").rglob("*.wav"), None)
            sample = work / "sample.wav"
            if source:
                with wave.open(str(source), "rb") as original:
                    params = original.getparams(); frames = original.readframes(30 * original.getframerate())
                with wave.open(str(sample), "wb") as out:
                    out.setparams(params); out.writeframes(frames)
                report["audio_source"] = str(source.relative_to(settings.root))
            else:
                with wave.open(str(sample), "wb") as out:
                    out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(b'\0\0' * 16000 * 3)
                report["audio_source"] = "Synthetic silence fixture; real speech accuracy not tested"
            segments, duration = engines.transcribe(sample)
            if source: assert segments, "No speech detected in real audio sample"
            return {"duration": duration, "segments": len(segments)}
        check("whisper", whisper)
        def vision():
            from PIL import Image
            with Image.open(image_path) as image:
                text, truncated = engines.describe(image.convert("RGB"), "Runtime verification.", brief=True)
            assert text and not truncated
            return {"characters": len(text)}
        check("vision", vision)
        engines.unload_vision()
        gc.collect()
    if require_gpu:
        cpu = [e for e in engines.events if e.get("device") == "cpu"]
        report["checks"]["required_gpu"] = {"passed": not cpu, "cpu_events": cpu}
    report["passed"] = all(item["passed"] for item in report["checks"].values())
    save_json(settings.root / "runtime/runtime_verification.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    main(parser.parse_args().require_gpu)
