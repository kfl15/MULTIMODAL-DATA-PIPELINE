# Modality — local multimodal preprocessing for RAG

Turn separate audio, text, document, spreadsheet, image, and video files into
separate readable text files. A local browser page lets you upload folders,
start processing, inspect progress, preview text, and download the results.

The application runs in Docker. **Ollama runs separately on your host.** Models
run locally; initial software/model downloads need internet access. This project
ends at text export: embeddings, vector storage, and RAG answering can follow later.

```text
input/audio/meeting.wav    ── Whisper ──────────────────────────┐
input/pdf/report.pdf      ── render → OCR → layout → reading ───┤
input/docs/report.docx    ── LibreOffice → document pipeline ───┤
input/image/figure.png    ── OCR + layout + VLM ────────────────┤
input/excel/budget.xlsx   ── cells + rendered visual analysis ──┤
input/video/session.mp4   ── speech + timestamped frames ───────┤
input/text/notes.txt      ── preserve text ─────────────────────┘
                                           ↓
                         output/<category>/<source-extension>.txt
```

## Requirements and initial downloads

Supported bootstrap targets: **Ubuntu 22.04/24.04 x86_64** and **Windows 11 x86_64**
with Docker Desktop's WSL2 backend. NVIDIA GPU is preferred; CPU mode is available
and slower. Allow at least **20 GiB for CPU or 40 GiB for GPU** in Docker storage
for images, downloads, and build caches. Source collections and outputs need
additional space in the project filesystem.

Setup installs Docker when missing and host Ollama, builds Python 3.10 and all
application dependencies into the Docker image, and downloads:

| Model | Purpose | Download location |
| --- | --- | --- |
| Whisper `tiny.en` | English speech | `models/whisper-tiny.en/` |
| PaddleOCR mobile detector + English recognizer | Image/page text | `models/paddle/` |
| PP-DocLayout-S | Region detection | `models/paddle/` |
| LayoutReader / LayoutLMv3 | Reading order | `models/layoutreader/` |
| `granite3.2-vision:2b` | Images, charts, tables | Host Ollama storage |

Model weights and datasets are **not in Git**. Downloads are reused. Setup skips
installed host software and skips the Ollama model pull when the model is already
installed. Local model files are verified by checksum and reused; Docker reuses
unchanged build layers. Host Python is left untouched because the application
uses Python inside Docker. See
[MODEL_NOTES.md](MODEL_NOTES.md) for revisions, attribution, licenses, and the
explicit GPU NCCL dependency exception. Python, FFmpeg, LibreOffice, and notebook
support are inside the image; you do not need to install host Python.

## Start on Linux

From this folder:

```bash
bash scripts/setup-linux.sh          # detects GPU; may request sudo for installation
# Or explicitly select the slower CPU build:
bash scripts/setup-linux.sh --cpu
```

Setup checks requirements, installs missing supported software, downloads and
verifies models, tests inference, and starts the page at **http://127.0.0.1:8000**.
NVIDIA drivers/firmware changes may require manual installation and a reboot.
If setup stops, follow its message and rerun; data and completed downloads remain.
If adding the NVIDIA runtime requires restarting Docker while other containers
are running, the script stops rather than interrupting those containers.

## Start on Windows

Run `scripts\setup-windows.bat` from File Explorer or Command Prompt. For CPU mode:

```bat
scripts\setup-windows.bat -Cpu
```

The launcher uses PowerShell and winget. Install/update Microsoft's App Installer
if winget is unavailable. Complete any administrator prompts, Docker Desktop
first-run steps, or WSL2 restart, then rerun the same file.

Use Docker Desktop's Linux containers with WSL2. Enable **Settings → Resources →
Network → Enable host networking**, then Apply/restart. The script tests that a
container can reach host Ollama without exposing Ollama on all interfaces.
Windows setup must be tested on a real Windows/WSL2 machine before being described
as verified. [Docker GPU requirements](https://docs.docker.com/desktop/features/gpu/),
[host networking](https://docs.docker.com/engine/network/drivers/host/).

## Your data and sample download

**Google Drive sample data: PENDING**

**`GOOGLE_DRIVE_DATA_LINK_HERE`**

<!-- DATA_DOWNLOAD_URL: paste the Google Drive folder link here -->

The owner will upload the data separately and replace that marker. No script
attempts to download from this placeholder. You can use your own data immediately.

The future download should contain an `input/` folder with these category folders:

```text
input/
  audio/   # WAV, MP3, M4A
  text/    # UTF-8 TXT, Markdown
  pdf/     # PDF
  excel/   # XLSX, CSV
  image/   # PNG, JPG/JPEG, TIF/TIFF
  video/   # MP4, MKV, MOV
  docs/    # DOCX
```

Place that `input/` directly inside this project, beside `compose.yaml`. Avoid
`input/input/`. Each category may contain many files and subfolders. Alternatively,
choose a category on the page and upload a folder or individual files. Unsupported
files are skipped by the browser with a count; invalid API uploads are rejected.

The current author's local inputs are copies of four meeting WAV recordings,
77 slides, the L6 report, and three distinct L9 document images. Original notebook
data paths are preserved. The bank-statement example contains existing annotations.
Alternative annotated views were not added as duplicate independent documents.
Previous JSONL exports stay under `reference_data/`; they are not raw video or new
model predictions. PDF, Office, and raw-video folders remain empty until supplied.

To reproduce this arrangement from the original workspace (optional, local Python
or a suitably mounted container):

```bash
python -m scripts.prepare_data --source-root ..
```

`runtime/data_manifest.json` records source paths, destinations, sizes, and checksums.
Neither these data files nor reference exports are tracked in Git.

## Processing and output

Select source categories, then click **Process selected folders**. Only one job
runs at a time to manage GPU memory. Upload after the active job finishes.
The app is intended for one local user and should stay bound to loopback.

Every source produces one UTF-8 text file, keeping its original extension:

```text
input/pdf/reports/annual.pdf → output/pdf/reports/annual.pdf.txt
input/image/annual.png      → output/image/annual.png.txt
```

Text/Markdown is preserved without model rewriting. PDFs and DOCX use rendered
pages, OCR, layout detection, reading order, and visual interpretation. Excel
exports sheet/cell references and formulas/cached values, then renders workbook
visuals. CSV is read directly. Audio includes speech timestamps; video includes
speech and frames sampled every 10 seconds by default. Silent videos still get
visual descriptions. Multipage TIFFs process all pages.

OCR/native text and **unverified VLM descriptions** are separate labeled sections.
The latter can contain mistakes, especially chart/table numbers. Failed later
stages keep usable earlier text and mark the file partial. Encrypted PDFs need
an unlocked copy. Pages over LayoutReader's 510-region limit retain OCR text with
a reading-order warning. No truncated regions or fabricated missing values.

Rerunning files replaces their current outputs atomically. Other outputs remain.
Jobs, device reports, errors, and OCR/layout coordinates persist in `runtime/jobs/`.
After a restart, unfinished jobs show as interrupted and can be explicitly rerun.
Download current text files or a selected completed job's ZIP from the page.

## Configuration and everyday commands

Setup writes `.env`. Start from `.env.example` for manual setup. `RUNTIME` selects
`cpu` or `gpu`; `DEVICE=auto` prefers GPU and `DEVICE=cpu` forces CPU. GPU mode also
uses `compose.gpu.yaml` (setup writes `COMPOSE_FILE` accordingly). Other settings:

- `OLLAMA_URL`: default `http://127.0.0.1:11434`, reached through host networking.
- `VISION_MODEL`: default `granite3.2-vision:2b`.
- `VIDEO_FRAME_INTERVAL`: default 10 seconds.
- `MAX_UPLOAD_MB`: per-file application limit, default 2048 MB.
- `LOCAL_UID` / `LOCAL_GID`: container file ownership; set to your Linux user/group IDs.

Keep host Ollama running. For an already configured checkout:

```bash
docker compose up -d
docker compose logs -f app
docker compose stop
# Rebuild after changing application code:
docker compose up -d --build
# Verify/reuse/download pinned models:
docker compose run --rm app python -m scripts.download_models
# Runtime checks and offline correctness tests:
docker compose run --rm app python -m scripts.verify_runtime
docker compose run --rm app python -m unittest discover -s tests -v
```

The health endpoint verifies the app is serving; **Check models & tools** tests
model files and host Ollama availability. Actual device execution appears in job
reports; merely detecting a GPU is not counted as successful GPU inference.
If the host GPU works but Docker rejects `--gpus`, install/configure NVIDIA Container
Toolkit and restart Docker as appropriate. CPU mode reports its fallback reason. The CPU image can use `DEVICE=auto` to let
host Ollama still use its GPU; `DEVICE=cpu` explicitly forces all inference to CPU.
Paddle MKLDNN is disabled to avoid the CPU OneDNN/PIR incompatibility found during
verification.

## Learning notebooks

Four notebooks call the same modules as the app, with comments and mappings to
original Lab 2/L6 cells. Pause the app while experimenting with GPU notebooks.

```bash
# Start notebook mode using the same image and mounted data:
docker compose -f compose.yaml -f compose.notebooks.yaml up
# GPU image: additionally include -f compose.gpu.yaml before the notebooks file.
```

Use the token URL printed by Jupyter on port 8888. Keep token authentication enabled.
Notebook files are mounted so edits persist. Stop notebook mode before restarting
the normal app with `docker compose up -d`.

## Repository and development

`app/` contains format adapters, model adapters, persistent jobs, and the FastAPI
API. `app/static/` is the browser page. The API reference is at `/docs`.
`scripts/` contains installers, pinned model downloads, checks, and local data copying.
There is no dependency on the parent course repository at runtime.

Commit code, documentation, empty notebook outputs, configuration, and folder
placeholders. `.gitignore` and `.dockerignore` exclude data, models, results,
reference exports, runtime state, local environments, and `.env`. Check what you
stage before publishing; never force-add the excluded datasets.

Real data only covers the formats present locally. Tests may create small temporary
PDF/Office/video fixtures; these are validation fixtures, not additional owner data.
See `VERIFICATION.md` for executed checks and known environment limitations.
