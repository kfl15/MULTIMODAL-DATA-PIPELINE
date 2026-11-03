# Verification record

Checked on 2026-11-03. These results describe the local development machine;
Windows and standard NVIDIA-container-runtime checks remain separate items below.

## Passed

- **Data preparation:** 91 original-to-copy mappings verified with SHA-256: 85 raw
  inputs and 6 reference files. Source data was copied; original notebook paths remain.
- **Offline/API tests:** all 25 tests pass inside the CPU Docker image. Coverage
  includes nested upload paths, invalid paths and symlinks, size limits, distinct
  output names, atomic reruns, historical ZIP snapshots, partial failures, continued
  batches, single-worker exclusion, and interrupted-job recovery.
- **Format adapters:** real LibreOffice conversion of temporary DOCX and XLSX/chart
  fixtures; real FFmpeg/ffprobe processing of temporary videos with and without audio;
  multipage PDFs/TIFFs and native spreadsheet/text handling. Those tests stub model
  calls and do not represent real-data accuracy measurements.
- **Real host GPU inference:** Torch calculation, Paddle OCR, PP-DocLayout-S,
  LayoutReader, Whisper on 30 seconds of real meeting speech, and Granite Vision.
  Actual inference devices/VRAM placement were recorded after execution.
- **Real CPU-container inference:** the same model stages, including host Ollama
  explicitly set to CPU for the check. Paddle's OneDNN/PIR failure was fixed by
  disabling MKLDNN for OCR/layout; the full corrected check passed.
- **Real source exports:** one complete 504.443-second meeting recording and the
  L6 report produced separate text files with no processing warnings. VLM content
  is still labeled unverified; successful execution does not establish correctness.
- **Docker builds:** CPU and GPU images build. CPU dependencies pass `pip check`.
  GPU dependencies have exactly the documented NCCL metadata exception, with no
  additional conflicts. Image sizes on this machine are about 5 GB (CPU) and
  17 GB (GPU), including shared layers; build/download caches need additional space.
- **Model downloads:** exercised a real pinned missing-model download (Paddle detector)
  and verified all remaining existing weights/checksums.
- **Browser QA:** page load, single-file upload, nested folder upload, job completion,
  text preview, ZIP download, and mobile width checks pass without page errors.
- **Notebooks/configuration:** four notebooks validate with clear outputs and
  compilable cells. Shell/JavaScript syntax and all three Compose configurations
  validate. Git ignore checks exclude data, weights, outputs, runtime state and
  environments while retaining code, notebooks and directory placeholders.
- **Local app startup:** the Compose app serves on `http://127.0.0.1:8000`, is healthy,
  and reports readiness with all model files, FFmpeg, ffprobe, LibreOffice and host
  Ollama available.

## Remaining platform checks

- This host's standard `docker run --gpus all` fails because NVIDIA Container
  Toolkit/runtime is not configured. Installing/configuring it requires the host's
  sudo password, which is not available to this session. The normal GPU Compose
  profile has therefore **not been verified running on GPU**. Host GPU checks are
  not represented as GPU-container checks.
- The running app uses the CPU image with `DEVICE=auto`: container OCR/reading
  order/speech fall back to CPU; host Ollama can still use GPU. For full container
  GPU execution, run `bash scripts/setup-linux.sh --gpu`, complete the reported
  runtime/restart steps, then rerun runtime verification.
- The Windows launcher is implemented but **not tested on Windows/WSL2**. Validate
  installation, required restarts, host networking, GPU passthrough, folder uploads,
  and downloads on a real Windows machine before describing it as verified.
- A clean-machine Linux installation of system packages was not performed: this
  host already had Docker/Ollama and requires a password for privileged changes.
  Both Docker images were built from their public base image/package downloads;
  the shell installer was syntax-checked and its application/model steps exercised.
- The owner's current data contains no raw PDFs, DOCX, XLSX or videos. Their adapters
  have fixture coverage; do not claim accuracy validation on nonexistent real inputs.

## Local evidence and repeatable commands

Evidence is in ignored `runtime/` files: `data_manifest.json`, `model_manifest.json`,
`host_gpu_verification.json`, `docker_cpu_verification.json`,
`real_source_verification.json`, `docker-offline-tests.log`, the Docker build logs,
and `browser-qa/` screenshots/JSON. These artifacts are not uploaded with the repo.

```bash
docker compose run --rm app python -m unittest discover -s tests -v
docker compose run --rm app python -m scripts.download_models --verify-only
docker compose run --rm app python -m scripts.verify_runtime
# After the NVIDIA container runtime is working, require actual GPU execution:
docker compose run --rm app python -m scripts.verify_runtime --require-gpu
```

The optional `scripts/browser_smoke.py` uses Playwright against a disposable app
root. It uploads synthetic test files and starts processing; do not point it at
unrelated active work. Runtime smoke checks may use explicitly labeled synthetic
fixtures when no source data is supplied.
