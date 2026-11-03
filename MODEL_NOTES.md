# Model sources and runtime notes

The preprocessing implementation adapts Lab 2's Whisper/Paddle code and L6's
OCR/layout/reading-order/vision stages. It does not import the parent project.
Pinned model repositories, revisions, and file checksums are in `models.lock.json`.

| Model | Source |
| --- | --- |
| Whisper tiny.en | https://huggingface.co/Systran/faster-whisper-tiny.en |
| OCR text detection | https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det |
| English OCR recognition | https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec |
| Document layout | https://huggingface.co/PaddlePaddle/PP-DocLayout-S |
| LayoutReader | https://huggingface.co/hantian/layoutreader |
| Granite vision | https://ollama.com/library/granite3.2-vision |

The LayoutReader checkpoint model card states **CC-BY-NC-SA-4.0**. Retain the
model card and consult its noncommercial/share-alike conditions before reuse.
Weights are downloaded separately, not included in this repository. Other models
retain their respective upstream licenses/model cards.

The unique reading-position assignment is adapted from FreeOCR-AI/LayoutReader:
https://github.com/FreeOCR-AI/layoutreader/blob/main/v3/helpers.py
The original implementation in this workspace is `tools/document_runtime.py`.
LayoutReader uses LayoutLMv3 for reading order; it does not classify the meaning
of every OCR line or reconstruct complete tables.

## Shared CUDA stack

The GPU image uses the original tested single-GPU stack: CUDA 12.6 pip runtime
libraries, Paddle 3.3.0, Torch 2.6.0+cu126, and NCCL 2.25.1. Torch declares an exact
NCCL 2.21.5 dependency. Installing Torch with `--no-deps` deliberately retains
Paddle's 2.25.1. This is one documented package-metadata conflict, not a clean
resolution of both requirements. `scripts.check_dependencies` rejects additional
conflicts. Multi-GPU/distributed behavior is not supported or tested.

The CPU build uses Paddle's CPU wheel and Torch's CPU wheel, without CUDA/NCCL.
OCR/layout explicitly disable MKLDNN: Paddle 3.3 otherwise raises an unsupported
OneDNN/PIR attribute error on this CPU environment. Real CPU inference was checked
after disabling it.

Inference runs sequentially, releasing models between stages. CUDA library paths
are prepared before Python starts by `scripts/entrypoint.sh`. Device reports
record real inference/backend placement and explicit fallback reasons. Ollama's
VRAM placement is recorded after generation; a nonzero allocation can still
mean partial CPU offload.

VLM descriptions are model-generated, unverified text. Keep them distinct from
native/OCR text and inspect the source before relying on numeric chart/table values.
The download script records the actual host Ollama model digest because Ollama
tags can change independently of pinned Hugging Face revisions.
