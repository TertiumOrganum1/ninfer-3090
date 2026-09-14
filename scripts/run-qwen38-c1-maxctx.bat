@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Qwen3.8-27B on one RTX 3090, tuned for context and prefix reuse rather than for caution.
rem
rem run-qwen38-c1.bat serves 65,536 tokens of INT8 and leaves 2.85 GiB of the card unused. This
rem profile spends it: rk8v4 KV, MTP3 speculation plus the draft head, and the tuned context cache.
rem
rem WHAT rk8v4 BUYS. The same KV that holds 171,648 INT8 tokens holds 226,560 rk8v4 tokens -- +33%
rem context for +0.082%% perplexity. It is opt-in precisely because INT8 is the quality default.
rem
rem CONTEXT CACHE. A checkpoint is a KV prefix plus a StateImage, and on this model the StateImage
rem is 147 MiB flat regardless of prefix length -- 48 GDN layers of 128x128x48 FP32 recurrent state
rem plus conv -- or 74.5 MiB with --gdn-state-fp16, which this profile uses. --host-state-slots 32
rem therefore pins 2.34 GiB of HOST memory rather than 4.59 GiB. It is what takes prefix reuse from
rem 8.4%% to 98.3%% on a multi-preamble workload.
rem
rem MEMORY FLAGS, both free on quality (docs/maintainer/quality-trade-experiments.md):
rem   --embedding-q4    token embedding stored as Q4 at load: -644 MiB of weights, perplexity
rem                     4.346413 -> 4.343738 (noise), decode unchanged.
rem   --gdn-state-fp16  recurrent state stored as FP16: -72 MiB per device state slot, perplexity
rem                     unchanged, greedy output bit-identical.
rem Together they buy one full rung: 163,840 now starts with the free memory 131,072 used to leave.
rem Adding --lm-head-q6 frees another 341 MiB (+12.9K tokens, +0.01%% perplexity) but costs 2-5%%
rem of single-user decode until a Q6 small-T kernel exists, so it is not on by default here.
rem
rem --auto-prefix-grid lets two callers whose prompts merely start alike share a cached prefix with
rem no client hint. A grid point is only materialised once two independent callers have both asked
rem for it, so it cannot waste a slot speculatively.
rem
rem MEASURED on this machine with the desktop running, which is the pessimistic case. Without the
rem two memory flags (the earlier profile):
rem
rem   lanes  KV      context   vision   runtime    free after startup
rem   ------------------------------------------------------------------
rem   1      int8     65,536   off      2.73 GiB   2.85 GiB   <- what run-qwen38-c1.bat does
rem   1      rk8v4   131,072   off      3.93 GiB   1.68 GiB
rem   1      rk8v4   131,072   overlay  3.94 GiB   1.59 GiB
rem   2      rk8v4   131,072   overlay  4.32 GiB   1.26 GiB
rem   1      rk8v4   163,840   overlay  4.78 GiB   763.2 MiB
rem
rem With --embedding-q4 --gdn-state-fp16 (2026-09-14, arms alternated at each rung, desktop holding
rem 1.25 GiB until 212,992, then 0.44 GiB):
rem
rem   context    without flags               with flags                  + --lm-head-q6
rem   ---------------------------------------------------------------------------------------
rem   131,072    3.94 GiB / 1.70 GiB free    3.80 GiB / 2.47 GiB free    2.79 GiB free
rem   163,840    4.79 GiB /  873 MiB free    4.65 GiB / 1.63 GiB free    1.96 GiB free  <- default
rem   196,608    5.63 GiB /    0 free        5.49 GiB /  798 MiB free    1.11 GiB free
rem   229,376    refused                     6.34 GiB /    0 free        276 MiB free
rem   245,760    refused                     refused                     starts, 0 free
rem
rem Windows keeps one lane by default: a desktop holds roughly 1.5 GiB of the card, so the
rem headroom above is what you actually have. Vision is on -- overlay residency costs about 10 MiB
rem of runtime reservation, so there is no reason to trade it away. run-qwen38-vision.bat remains
rem for the plain 32K image profile.
rem Rungs if startup refuses: 196608 / 163840 / 131072 / 114688 / 98304 / 65536.
rem ---------------------------------------------------------------------------------------------

rem Every setting below can be overridden from the environment without editing this file:
rem
rem   set NINFER_HOST=0.0.0.0 && run-qwen38-c1-maxctx.bat
rem
rem The default model path matches what download-qwen38-27b.bat writes and how the release archive
rem is laid out: this launcher sits beside models\.
set "MODEL=%~dp0models\qwen3_8_27b.ninfer"
set "CONTEXT=163840"
set "CONCURRENCY=1"
rem Loopback by default. 0.0.0.0 publishes an unauthenticated OpenAI-compatible endpoint to every
rem network this machine is on, so it is opt-in per run rather than the shipped default.
set "HOST=127.0.0.1"
set "PORT=8080"
set "KV_DTYPE=rk8v4"

if not "%NINFER_MODEL%"=="" set "MODEL=%NINFER_MODEL%"
if not "%NINFER_CONTEXT%"=="" set "CONTEXT=%NINFER_CONTEXT%"
if not "%NINFER_CONCURRENCY%"=="" set "CONCURRENCY=%NINFER_CONCURRENCY%"
if not "%NINFER_HOST%"=="" set "HOST=%NINFER_HOST%"
if not "%NINFER_PORT%"=="" set "PORT=%NINFER_PORT%"
if not "%NINFER_KV_DTYPE%"=="" set "KV_DTYPE=%NINFER_KV_DTYPE%"

set "ROOT=%~dp0.."
set "SERVER=%ROOT%\build-ninja\apps\ninfer-serve.exe"
if not exist "%SERVER%" set "SERVER=%~dp0ninfer-serve.exe"
if not "%NINFER_SERVER%"=="" set "SERVER=%NINFER_SERVER%"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Build it first:  .\scripts\build.ps1
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Download it first:  download-qwen38-27b.bat
  exit /b 1
)

echo Qwen3.8-27B  ^|  C%CONCURRENCY%  ^|  context %CONTEXT%  ^|  rk8v4 KV  ^|  MTP3 + draft head  ^|  Vision (overlay)
echo Memory: Q4 token embedding, FP16 GDN state
echo Cache: 8 shared / 8 private / 32 host states  ^|  automatic prefix grid on
echo API: http://%HOST%:%PORT%/v1
echo.

rem WHAT --host-kv-mib 8192 ACTUALLY GETS ON WINDOWS, which is not 8 GiB. WDDM maps a pinned host
rem allocation into the GPU's address space and charges it against the card, so the runtime clamps
rem the request to (free VRAM - 1 GiB) / 2 before the first cudaMallocHost -- it cannot ask and back
rem off, because one failure poisons every later attempt in the process. At this profile's measured
rem residency that resolves to:
rem
rem   launcher                        free after startup   pinned host KV
rem   -------------------------------------------------------------------
rem   run-qwen38-c1-maxctx (this)           1.59 GiB           302 MiB
rem   run-qwen36-35b-a3b-c1-maxctx        184-344 MiB        0 -- none at all
rem
rem The flag is kept rather than corrected because it is right on the .sh launchers, where Linux
rem pins the full 8 GiB of host RAM, and because it is harmless here: the clamp takes what is
rem actually free after the KV cache is allocated, so it costs no context, and prefix reuse falls
rem back to device pages when the pin is zero. Do not read "8192" as a description of this machine.

"%SERVER%" "%MODEL%" ^
  --host %HOST% --port %PORT% ^
  --max-concurrency %CONCURRENCY% ^
  --max-context %CONTEXT% ^
  --kv-capacity %CONTEXT% ^
  --kv-dtype %KV_DTYPE% ^
  --spec mtp --draft-tokens 3 --lm-head-draft ^
  --embedding-q4 --gdn-state-fp16 ^
  --prefill-chunk 1024 ^
  --max-pending-requests 16 --pending-timeout-ms 600000 ^
  --vision --vision-residency overlay ^
  --max-private-continuations 8 --max-shared-prefixes 8 --host-state-slots 32 --host-kv-mib 8192 ^
  --auto-prefix-grid

endlocal
