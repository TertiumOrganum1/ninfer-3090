# NInfer-3090 for Windows — release archive

This is the README that ships **inside** the Windows release archive, where every file sits in one
directory. If you are reading it in a checkout instead, the launchers and downloaders it names live
under `scripts/`, and
[docs/rtx-3090-windows.md](https://github.com/ashalliants/ninfer-3090/blob/master/docs/rtx-3090-windows.md)
is the fuller guide. Links here are absolute on purpose: the archive ships no `docs/` directory.

Check `VERSION` for the release this archive was cut from, and `RELEASE_NOTES_*.md` for what
changed.

## Requirements

- Windows 11 x64
- GeForce RTX 3090 (or 3090 Ti) with a recent NVIDIA driver
- Microsoft Visual C++ 2022 runtime

Model artifacts are **not** included — they are 17–21 GB each. The downloaders below fetch them.

## Quick start

From the directory you unpacked, two commands (or double-click either file and pick a model):

```powershell
.\download-model.bat qwen36-35b-a3b    # ~21 GB, resumable, verifies size and SHA256
.\run.bat qwen36-35b-a3b               # serves on http://127.0.0.1:8080/v1
```

The downloader writes into `models\` beside these files, which is where the launcher looks.
`NINFER_MODEL_DIR` moves where the **downloader** puts artifacts; to point the **launcher** at a file
somewhere else, give it `NINFER_MODEL` (the launcher does not read `NINFER_MODEL_DIR`).

That is the recommended Windows profile: Qwen3.6-35B-A3B with `rk8v4` KV, MTP3 speculation plus the
draft head, and vision in overlay residency. For the dense 27B instead:

```powershell
.\download-model.bat qwen38-27b        # ~19 GB, the DFlash2 bundle; it also carries the MTP weights
.\run.bat qwen38-27b                   # one user, 131,072 tokens, DFlash2 (fastest)
```

For the longest context instead, `set NINFER_SPEC=mtp && .\run.bat qwen38-27b` runs the MTP profile at
163,840 tokens with a smaller prefill chunk and `--lm-head-q6`: slower decode, more context.

`run.bat qwen38-27b int8` and `run.bat qwen38-27b c8` are the older INT8 profiles — one user at
65,536 tokens, and eight concurrent users at 8,192 each.

The endpoint is OpenAI-compatible, so anything that speaks `/v1/chat/completions` works. Leave the
API key blank.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve.exe` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer.exe` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench.exe` | throughput benchmark against the public Engine route |
| `download-model.bat` | pinned, resumable artifact downloads with verification |
| `run.bat` | serving profiles: `run.bat <model> [profile]`, or double-click it to pick a model |
| `*.dll` | the FFmpeg and libcurl runtime dependencies |
| `SHA256SUMS.txt` | checksums for every file in this directory |

## Overrides

Nothing here needs editing. **Every** profile reads `NINFER_MODEL`, `NINFER_SERVER`, `NINFER_HOST`
and `NINFER_PORT`. The default (`tuned`) profiles read more:

| profile | also reads |
|---|---|
| `run.bat qwen38-27b` | `NINFER_CONTEXT`, `NINFER_CONCURRENCY`, `NINFER_KV_DTYPE`, `NINFER_SPEC` (`dflash2`, `mtp`, `none`), `NINFER_DRAFT_TOKENS`, `NINFER_PREFILL_CHUNK`, `NINFER_VISION`, `NINFER_VISION_RESIDENCY` |
| `run.bat qwen36-35b-a3b` | the same, except `NINFER_SPEC` is `mtp` or `none` |
| `run.bat qwen38-27b int8`, `run.bat qwen38-27b c8` | nothing further; every serving flag is fixed |
| `download-model.bat <model>` | `NINFER_MODEL_DIR` |

> `NINFER_HOST=0.0.0.0` exposes the server to your network **unauthenticated**. The launchers bind
> `127.0.0.1` for that reason. If you set it, put something in front of it.

## If startup refuses

The message names the numbers. A 24 GB card running a desktop has roughly 1.5 GiB less to work
with than a headless one, so the largest profiles do not fit alongside a desktop:

```
requested Engine runtime reservation requires 2864526592 bytes,
but only 2375691264 bytes are available for runtime capacity
```

Drop a context rung first: set `NINFER_CONTEXT` to the next rung below the profile's default; the
rungs are listed in the `run.bat` header, measured on this card — for
`run.bat qwen38-27b` (default 131,072 with DFlash2, 163,840 with `NINFER_SPEC=mtp`) that is
114688, then 98304, then 81920. Speculation is the next lever (`NINFER_SPEC=none`), worth about
992 MiB on the 35B-A3B at the cost of decode speed.
Drop vision last: in overlay residency it costs almost nothing resident.

To size a profile before running it, open
[docs/config-calculator.html](https://github.com/ashalliants/ninfer-3090/blob/master/docs/config-calculator.html)
from the repository — one self-contained file, no network needed, every constant in it measured on
an RTX 3090. It is not in this archive.

## Full documentation

<https://github.com/ashalliants/ninfer-3090>
