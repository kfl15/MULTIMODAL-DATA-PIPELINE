"""Format adapters: one source file in, one text Result out.

Heavy libraries are imported only by the adapter that needs them. The same
Processor is used by the API and by all four teaching notebooks.
"""
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from .config import FORMATS
from .engines import Engines
from .results import Result


def command(arguments, timeout=600):
    """Use argument arrays: source filenames are never interpreted by a shell."""
    result = subprocess.run([str(a) for a in arguments], capture_output=True,
                            text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{arguments[0]} failed: {result.stderr[-2000:]}")
    return result.stdout


def timestamp(seconds):
    hours, remainder = divmod(float(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"


class Processor:
    def __init__(self, settings, engines=None, progress=None):
        self.settings = settings
        self.engines = engines or Engines(settings)
        self.progress = progress or (lambda message: None)

    def process(self, path, category):
        path = Path(path)
        if category not in FORMATS or path.suffix.lower() not in FORMATS[category]:
            raise ValueError(f"Unsupported {category} source: {path.name}")
        event_start = len(self.engines.events)
        self.progress(f"Reading {category} source")
        with tempfile.TemporaryDirectory(prefix="processing-", dir=self.settings.root / "runtime") as work:
            work = Path(work)
            if category == "text":
                result = Result(sections=[path.read_text(encoding="utf-8-sig")])
            elif category == "excel":
                result = self.spreadsheet(path, work)
            elif category == "audio":
                result = self.audio(path, work)
            elif category == "video":
                result = self.video(path, work)
            elif category == "docs":
                self.progress("Rendering Word document with LibreOffice")
                result = self.document(self.office_pdf(path, work), work)
            else:
                result = self.document(path, work)
        result.devices.extend(self.engines.events[event_start:])
        if not any(section.strip() for section in result.sections):
            result.warnings.append("No content was extracted from this source.")
        return result

    def office_pdf(self, source, work):
        directory = work / "office"
        directory.mkdir(exist_ok=True)
        profile = (work / "libreoffice-profile").as_uri()
        command(["libreoffice", f"-env:UserInstallation={profile}", "--headless", "--convert-to",
                 "pdf", "--outdir", directory, source], timeout=300)
        target = directory / (source.stem + ".pdf")
        if not target.is_file():
            raise RuntimeError("LibreOffice did not produce a PDF; check whether the document is supported or encrypted")
        return target

    def spreadsheet(self, source, work):
        result = Result()
        if source.suffix.lower() == ".csv":
            with source.open(encoding="utf-8-sig", newline="") as handle:
                sample = handle.read(8192)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    dialect = csv.excel
                for i, row in enumerate(csv.reader(handle, dialect), 1):
                    result.sections.append(f"Row {i}: " + json.dumps(row, ensure_ascii=False))
            return result
        import openpyxl
        self.progress("Reading spreadsheet cells and formulas")
        formulas = openpyxl.load_workbook(source, data_only=False, read_only=True)
        cached = openpyxl.load_workbook(source, data_only=True, read_only=True)
        try:
            for sheet in formulas:
                result.sections.append(f"## Sheet: {sheet.title}")
                for row in sheet:
                    values = []
                    for cell in row:
                        if cell.value is None:
                            continue
                        value = str(cell.value)
                        if cell.data_type == "f":
                            cache = cached[sheet.title][cell.coordinate].value
                            value += f" [cached value: {cache if cache is not None else 'unavailable'}]"
                        values.append(f"{cell.coordinate}={value}")
                    if values:
                        result.sections.append(" | ".join(values))
        finally:
            formulas.close()
            cached.close()
        try:
            self.progress("Rendering workbook charts and images")
            rendered = self.office_pdf(source, work)
            result.sections.append("## Workbook visual analysis (rendered page references; sheet association not inferred)")
            result.merge(self.document(rendered, work, full_page_vision=True))
        except Exception as error:
            result.warnings.append(f"Spreadsheet visual analysis failed; cell content retained: {error}")
        return result

    def audio(self, source, work):
        self.progress("Preparing audio with FFmpeg")
        prepared = work / "speech.wav"
        command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", source,
                 "-vn", "-ac", "1", "-ar", "16000", prepared])
        self.progress("Transcribing speech with Whisper")
        segments, duration = self.engines.transcribe(prepared)
        result = Result(sections=[f"## Speech transcript\nDuration: {duration:.3f} seconds"])
        for segment in segments:
            result.sections.append(f"[{timestamp(segment['start'])} --> {timestamp(segment['end'])}] {segment['text']}")
        result.details.append({"kind": "speech", "duration": duration, "segments": segments})
        if not segments:
            result.sections.append("No speech was recognized.")
        return result

    def document(self, source, work, full_page_vision=False):
        from PIL import Image, ImageOps, ImageSequence
        result = Result()
        self.engines.unload_vision()
        if source.suffix.lower() == ".pdf":
            import pymupdf
            with pymupdf.open(source) as pdf:
                if pdf.needs_pass:
                    raise ValueError("Password-protected PDF: provide an unlocked copy")
                for i, page in enumerate(pdf, 1):
                    try:
                        self.progress(f"Rendering page {i}/{len(pdf)}")
                        target = work / f"page-{i}.png"
                        page.get_pixmap(dpi=144, alpha=False).save(target)
                        native = page.get_text().strip()
                        if native:
                            result.sections.append(f"## Page {i}: native PDF text\n{native}")
                        with Image.open(target) as image:
                            result.merge(self.image_page(image, target, f"Page {i}", full_page_vision))
                    except Exception as error:
                        result.warnings.append(f"Page {i} failed: {error}")
        else:
            with Image.open(source) as image:
                for i, frame in enumerate(ImageSequence.Iterator(image), 1):
                    target = work / f"image-page-{i}.png"
                    prepared = ImageOps.exif_transpose(frame).convert("RGB")
                    prepared.save(target)
                    result.merge(self.image_page(prepared, target, f"Page {i}", full_page_vision))
        return result

    def image_page(self, image, path, reference, full_page_vision=False):
        result = Result()
        self.engines.unload_vision()
        regions, layout = [], []
        self.progress(f"{reference}: PaddleOCR")
        try:
            regions = self.engines.ocr(path)
        except Exception as error:
            result.warnings.append(f"{reference}: OCR failed: {error}")
        self.progress(f"{reference}: layout detection")
        try:
            layout = self.engines.layout(path)
        except Exception as error:
            result.warnings.append(f"{reference}: layout detection failed: {error}")
        indexes = list(range(len(regions)))
        if regions:
            self.progress(f"{reference}: LayoutReader reading order")
            try:
                indexes = self.engines.order(regions, image.width, image.height)
            except Exception as error:
                result.warnings.append(f"{reference}: reading order unavailable; original OCR order retained: {error}")
        ordered_regions = [regions[i] for i in indexes]
        ocr_text = "\n".join(region["text"] for region in ordered_regions)
        result.sections.append(f"## {reference}: OCR text\n" + (ocr_text or "No text recognized."))
        visuals = [region for region in layout if region['label'].lower()
                   in {"image", "figure", "chart", "table", "picture", "illustration"}]
        if full_page_vision or not visuals:
            visuals = [{"label": "full page/image", "bbox": [0, 0, image.width, image.height]}]
        for i, visual in enumerate(visuals, 1):
            box = visual["bbox"]
            box = [max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3])]
            if box[2] <= box[0] or box[3] <= box[1]:
                result.warnings.append(f"{reference}: invalid visual crop {i}")
                continue
            # Give the model text from inside the crop, not unrelated neighboring regions.
            contextual = "\n".join(r["text"] for r in ordered_regions
                if box[0] <= (r["bbox"][0] + r["bbox"][2]) / 2 <= box[2]
                and box[1] <= (r["bbox"][1] + r["bbox"][3]) / 2 <= box[3])
            self.progress(f"{reference}: VLM {visual['label']} {i}/{len(visuals)}")
            try:
                description, truncated = self.engines.describe(image.crop(box), contextual, visual["label"])
                result.sections.append(f"### {reference}: {visual['label']} {i} — VLM description (unverified)\n{description}")
                if truncated:
                    result.warnings.append(f"{reference}: VLM description {i} was truncated")
            except Exception as error:
                result.warnings.append(f"{reference}: VLM description {i} failed; OCR retained: {error}")
        result.details.append({"reference": reference, "width": image.width, "height": image.height,
                               "ocr": regions, "layout": layout, "reading_indexes": indexes})
        return result

    def video(self, source, work):
        result = Result()
        metadata = json.loads(command(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", source]))
        streams = metadata.get("streams", [])
        if any(s.get("codec_type") == "audio" for s in streams):
            try:
                result.merge(self.audio(source, work))
            except Exception as error:
                result.warnings.append(f"Video audio transcription failed: {error}")
        else:
            result.sections.append("## Audio\nThis video has no audio track.")
        if not any(s.get("codec_type") == "video" for s in streams):
            result.warnings.append("No video stream found")
            return result
        duration = float(metadata.get("format", {}).get("duration", 0))
        if duration <= 0:
            result.warnings.append("Video duration unavailable; unable to sample frames")
            return result
        interval = self.settings.frame_interval
        result.sections.append(f"## Video visuals\nFrames sampled every {interval:g} seconds; not every frame is inspected.")
        from PIL import Image
        current = 0.0
        while current < duration:
            frame = work / "video-frame.png"
            try:
                self.progress(f"Sampling video at {timestamp(current)}")
                frame.unlink(missing_ok=True)
                command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(current),
                         "-i", source, "-frames:v", "1", frame], timeout=120)
                with Image.open(frame) as image:
                    result.merge(self.image_page(image.convert("RGB"), frame,
                                                 f"Video frame near {timestamp(current)}", True))
            except Exception as error:
                result.warnings.append(f"Frame near {timestamp(current)} failed: {error}")
            current += interval
        return result
