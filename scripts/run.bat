@echo off
setlocal
rem ---------------------------------------------------------------------------------------------
rem Serve a model on one RTX 3090.
rem
rem   run.bat [model] [profile]     (double-click it and it asks for the model)
rem
rem   model             profiles
rem   qwen38-27b        tuned (default), int8, c8
rem   qwen36-35b-a3b    tuned (default)
rem
rem `tuned` is the recommended profile: rk8v4 KV, speculation plus the draft head, the memory
rem flags, vision in overlay residency, and the tuned context cache with automatic prefix grid.
rem `int8` and `c8` are the older reference profiles for the 27B -- one user at 64K of INT8 KV
rem (the quality default), and eight lanes at 8K -- with every serving flag fixed.
rem
rem Every measurement behind these defaults, the memory model, and the reasoning for each flag are
rem in docs\maintainer\launcher-profiles.md. What follows is what you need to run it.
rem
rem QWEN3.8-27B, `tuned`: two flag sets, each measured (docs\performance.md, "Recommended
rem configurations"), chosen with NINFER_SPEC. The default is the fast one.
rem
rem   NINFER_SPEC=dflash2 (default): fastest at one stream, for context up to about 130K
rem
rem     --spec dflash2 --draft-tokens 7 --lm-head-draft
rem     --prefill-cublas --prefill-chunk 4096
rem     --kv-dtype rk8v4 --embedding-q4 --gdn-state-fp16
rem     --vision --vision-residency overlay
rem
rem   NINFER_SPEC=mtp: the longest context, still fast -- 200,000 tokens verified by loading it
rem
rem     --spec mtp --draft-tokens 3 --lm-head-draft
rem     --prefill-cublas --prefill-chunk 2048
rem     --kv-dtype rk8v4 --embedding-q4 --lm-head-q6 --gdn-state-fp16
rem     --vision --vision-residency overlay
rem
rem   set NINFER_SPEC=mtp && run.bat qwen38-27b
rem
rem Against the previous defaults the DFlash2 set is about 1.7x on prefill and 1.39x on decode, for
rem +0.156%% perplexity from the cuBLAS route and +0.083%% from rk8v4. It gives up context because
rem DFlash2's draft weights and its refusal of --lm-head-q6 cost about 65K tokens between them: the
rem same flags on DFlash2 load at 130K and fail at 150K. `none` is the mtp set without speculation.
rem The qwen3_8_27b.ninfer that download-model.bat fetches is the DFlash2 bundle and carries the
rem MTP weights too, so one file serves both.
rem
rem OVERRIDES, from the environment. All profiles: NINFER_MODEL (artifact path), NINFER_SERVER,
rem NINFER_HOST, NINFER_PORT. `tuned` also: NINFER_CONTEXT, NINFER_CONCURRENCY, NINFER_KV_DTYPE,
rem NINFER_SPEC, NINFER_DRAFT_TOKENS, NINFER_PREFILL_CHUNK, NINFER_VISION (on^|off),
rem NINFER_VISION_RESIDENCY. Windows keeps one lane by default: a desktop holds roughly 1.5 GiB of
rem the card. Each spec's defaults (context, chunk) are the ones that fit; if startup refuses, drop
rem a rung of NINFER_CONTEXT: 196608 / 163840 / 131072 / 114688 / 98304 / 65536.
rem
rem Loopback by default. 0.0.0.0 publishes an unauthenticated OpenAI-compatible endpoint to every
rem network this machine is on, so it is opt-in per run rather than the shipped default:
rem
rem   set NINFER_HOST=0.0.0.0 && run.bat qwen38-27b
rem ---------------------------------------------------------------------------------------------

set "MODEL_KEY=%~1"
set "PROFILE=%~2"
if "%PROFILE%"=="" set "PROFILE=tuned"
if "%MODEL_KEY%"=="" goto :choose_model
:model_resolve
if /i "%MODEL_KEY%"=="-h" goto :help
if /i "%MODEL_KEY%"=="--help" goto :help
if /i "%MODEL_KEY%"=="qwen38-27b" (
  set "ARTIFACT=qwen3_8_27b.ninfer"
  set "TITLE=Qwen3.8-27B"
  goto :model_known
)
if /i "%MODEL_KEY%"=="qwen36-35b-a3b" (
  set "ARTIFACT=qwen3_6_35b_a3b.ninfer"
  set "TITLE=Qwen3.6-35B-A3B"
  goto :model_known
)
echo Unknown model: %MODEL_KEY% 1>&2
call :usage 1>&2
exit /b 2

:choose_model
rem Double-clicked from Explorer there is no argument to give, so ask. choice exits 255 when it has
rem no console to read from; that must not silently pick a model.
echo Which model?
echo   1  Qwen3.6-35B-A3B  (recommended)
echo   2  Qwen3.8-27B
choice /c 12 /n /m "Choose 1 or 2: "
if errorlevel 255 exit /b 2
if errorlevel 2 (
  set "MODEL_KEY=qwen38-27b"
) else (
  set "MODEL_KEY=qwen36-35b-a3b"
)
goto :model_resolve

:help
call :usage
exit /b 0

:model_known
rem The default model path matches what download-model.bat writes and how the release archive is
rem laid out: this launcher sits beside models\.
set "MODEL=%~dp0models\%ARTIFACT%"
set "HOST=127.0.0.1"
set "PORT=8080"
set "KV_DTYPE=rk8v4"
if not "%NINFER_MODEL%"=="" set "MODEL=%NINFER_MODEL%"
if not "%NINFER_HOST%"=="" set "HOST=%NINFER_HOST%"
if not "%NINFER_PORT%"=="" set "PORT=%NINFER_PORT%"

set "ROOT=%~dp0.."
set "SERVER=%ROOT%\build-ninja\apps\ninfer-serve.exe"
if not exist "%SERVER%" set "SERVER=%~dp0ninfer-serve.exe"
if not "%NINFER_SERVER%"=="" set "SERVER=%NINFER_SERVER%"

rem The profile fixes the whole serving shape. LABEL is the banner; PROFILE_ARGS is everything
rem after --host/--port. Values below use ^| for the separator: a bare pipe inside an expanded
rem variable would be parsed as a pipe when echoed.
set "LABEL="
set "PROFILE_ARGS="
set "PREFILL_NOTE="
set "HINT="
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/tuned" goto :profile_27b_tuned
if /i "%MODEL_KEY%/%PROFILE%"=="qwen36-35b-a3b/tuned" goto :profile_35b_tuned
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/int8" goto :profile_27b_int8
if /i "%MODEL_KEY%/%PROFILE%"=="qwen38-27b/c8" goto :profile_27b_c8
echo Model %MODEL_KEY% has no profile %PROFILE% 1>&2
call :usage 1>&2
exit /b 2

:profile_27b_tuned
rem The speculative backend fixes everything that has to move with it: the prefill chunk (the
rem cuBLAS route's workspace scales with it), the context that fits, and --lm-head-q6, which
rem DFlash and DFlash2 refuse. The overrides are applied after these defaults, so an explicit
rem NINFER_CONTEXT or NINFER_PREFILL_CHUNK always wins.
set "SPEC=dflash2"
if not "%NINFER_SPEC%"=="" set "SPEC=%NINFER_SPEC%"
if /i "%SPEC%"=="dflash2" goto :spec_dflash2
if /i "%SPEC%"=="mtp" goto :spec_mtp
if /i "%SPEC%"=="none" goto :spec_none
echo NINFER_SPEC must be dflash2, mtp or none, got %SPEC% 1>&2
exit /b 2

:spec_dflash2
set "SPEC=dflash2"
set "CONTEXT=131072"
set "PREFILL_CHUNK=4096"
set "DRAFT_TOKENS=7"
set "MEMORY_ARGS="
goto :spec_done

:spec_mtp
set "SPEC=mtp"
set "CONTEXT=163840"
set "PREFILL_CHUNK=2048"
set "DRAFT_TOKENS=3"
set "MEMORY_ARGS=--lm-head-q6"
goto :spec_done

:spec_none
set "SPEC=none"
set "CONTEXT=163840"
set "PREFILL_CHUNK=2048"
set "DRAFT_TOKENS="
set "MEMORY_ARGS="

:spec_done
set "CONCURRENCY=1"
if not "%NINFER_DRAFT_TOKENS%"=="" set "DRAFT_TOKENS=%NINFER_DRAFT_TOKENS%"
if not "%NINFER_CONTEXT%"=="" set "CONTEXT=%NINFER_CONTEXT%"
if not "%NINFER_CONCURRENCY%"=="" set "CONCURRENCY=%NINFER_CONCURRENCY%"
if not "%NINFER_KV_DTYPE%"=="" set "KV_DTYPE=%NINFER_KV_DTYPE%"
if not "%NINFER_PREFILL_CHUNK%"=="" set "PREFILL_CHUNK=%NINFER_PREFILL_CHUNK%"
set "SPEC_ARGS="
if /i not "%SPEC%"=="none" set "SPEC_ARGS=--spec %SPEC% --draft-tokens %DRAFT_TOKENS% --lm-head-draft"
if /i "%SPEC%"=="dflash2" set "SPEC_LABEL=DFlash2 K=%DRAFT_TOKENS% + draft head"
if /i "%SPEC%"=="mtp" set "SPEC_LABEL=MTP%DRAFT_TOKENS% + draft head, Q6 head, longest context"
if /i "%SPEC%"=="none" set "SPEC_LABEL=no speculation"
set "PROFILE_ARGS=--max-concurrency %CONCURRENCY% --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-dtype %KV_DTYPE% %SPEC_ARGS% --embedding-q4 %MEMORY_ARGS% --gdn-state-fp16 --prefill-cublas --prefill-chunk %PREFILL_CHUNK%"
set "LABEL=C%CONCURRENCY%  ^|  context %CONTEXT%  ^|  %KV_DTYPE% KV  ^|  %SPEC_LABEL%"
set "PREFILL_NOTE=Prefill: cuBLAS route, chunk %PREFILL_CHUNK%"
if /i "%SPEC%"=="dflash2" set "HINT=Need more than 130K context? Set NINFER_SPEC=mtp: longer context, slower decode."
goto :profile_tuned_common

:profile_35b_tuned
set "SPEC=mtp"
if not "%NINFER_SPEC%"=="" set "SPEC=%NINFER_SPEC%"
set "CONTEXT=147456"
set "CONCURRENCY=1"
set "PREFILL_CHUNK=512"
set "DRAFT_TOKENS=3"
if /i "%SPEC%"=="mtp" goto :spec35_mtp
if /i "%SPEC%"=="none" goto :spec35_none
echo NINFER_SPEC must be mtp or none, got %SPEC% 1>&2
exit /b 2
:spec35_mtp
set "SPEC=mtp"
set "SPEC_LABEL=MTP3 + draft head"
goto :spec35_done
:spec35_none
set "SPEC=none"
set "SPEC_LABEL=no speculation"
:spec35_done
if not "%NINFER_DRAFT_TOKENS%"=="" set "DRAFT_TOKENS=%NINFER_DRAFT_TOKENS%"
if not "%NINFER_CONTEXT%"=="" set "CONTEXT=%NINFER_CONTEXT%"
if not "%NINFER_CONCURRENCY%"=="" set "CONCURRENCY=%NINFER_CONCURRENCY%"
if not "%NINFER_KV_DTYPE%"=="" set "KV_DTYPE=%NINFER_KV_DTYPE%"
if not "%NINFER_PREFILL_CHUNK%"=="" set "PREFILL_CHUNK=%NINFER_PREFILL_CHUNK%"
set "SPEC_ARGS="
if /i "%SPEC%"=="mtp" set "SPEC_ARGS=--spec mtp --draft-tokens %DRAFT_TOKENS% --lm-head-draft --mtp-experts-q4"
if /i "%SPEC%"=="mtp" set "SPEC_LABEL=MTP%DRAFT_TOKENS% + draft head"
set "PROFILE_ARGS=--max-concurrency %CONCURRENCY% --max-context %CONTEXT% --kv-capacity %CONTEXT% --kv-dtype %KV_DTYPE% %SPEC_ARGS% --gdn-state-fp16 --prefill-chunk %PREFILL_CHUNK%"
set "LABEL=C%CONCURRENCY%  ^|  context %CONTEXT%  ^|  %KV_DTYPE% KV  ^|  %SPEC_LABEL%"
goto :profile_tuned_common

:profile_27b_int8
set "PROFILE_ARGS=--max-context 65536 --kv-capacity 65536 --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 --prefill-chunk 1024 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft"
set "LABEL=one request  ^|  64K context  ^|  INT8 KV  ^|  MTP3, ReplaySSM"
goto :launch

:profile_27b_c8
set "PROFILE_ARGS=--max-context 8192 --kv-capacity 16384 --max-concurrency 8 --max-pending-requests 32 --pending-timeout-ms 600000 --prefill-chunk 512 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft"
set "LABEL=up to eight requests  ^|  8K context  ^|  INT8 KV  ^|  MTP3, ReplaySSM"
goto :launch

:profile_tuned_common
rem Only `tuned` carries the context cache and vision: the reference profiles are deliberately
rem minimal. Vision stays on -- overlay residency keeps the tower host-pinned and streams each
rem image through a borrowed device window, so it costs about 10 MiB of runtime reservation.
set "VISION=on"
if not "%NINFER_VISION%"=="" set "VISION=%NINFER_VISION%"
set "VISION_RESIDENCY=overlay"
if not "%NINFER_VISION_RESIDENCY%"=="" set "VISION_RESIDENCY=%NINFER_VISION_RESIDENCY%"
set "VISION_ARGS="
if /i "%VISION%"=="on" (
  set "VISION_ARGS=--vision --vision-residency %VISION_RESIDENCY%"
  set "LABEL=%LABEL%  ^|  vision (overlay)"
  goto :vision_done
)
if /i "%VISION%"=="off" (
  set "LABEL=%LABEL%  ^|  text only"
  goto :vision_done
)
echo NINFER_VISION must be on or off, got %VISION% 1>&2
exit /b 2
:vision_done
set "PROFILE_ARGS=%PROFILE_ARGS% --max-pending-requests 16 --pending-timeout-ms 600000 %VISION_ARGS% --max-private-continuations 8 --max-shared-prefixes 8 --host-state-slots 32 --host-kv-mib 8192 --auto-prefix-grid"

:launch
if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Build it first:  .\scripts\build.ps1
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Download it first:  download-model.bat %MODEL_KEY%
  exit /b 1
)

echo %TITLE%  ^|  %LABEL%
if not "%PREFILL_NOTE%"=="" echo %PREFILL_NOTE%
if /i "%PROFILE%"=="tuned" echo Cache: 8 shared / 8 private / 32 host states  ^|  automatic prefix grid on
if not "%HINT%"=="" echo %HINT%
echo API: http://%HOST%:%PORT%/v1
echo.

rem WHAT --host-kv-mib 8192 ACTUALLY GETS ON WINDOWS, which is not 8 GiB. WDDM maps a pinned host
rem allocation into the GPU's address space and charges it against the card, so the runtime clamps
rem the request to (free VRAM - 1 GiB) / 2 before the first cudaMallocHost. The flag is kept
rem rather than corrected because it is right on Linux, where the full 8 GiB of host RAM is pinned,
rem and because it is harmless here: the clamp takes what is actually free after the KV cache is
rem allocated, so it costs no context, and prefix reuse falls back to device pages when the pin is
rem zero. Do not read "8192" as a description of this machine. See
rem docs\maintainer\launcher-profiles.md.
"%SERVER%" "%MODEL%" --host %HOST% --port %PORT% %PROFILE_ARGS%
endlocal
exit /b %ERRORLEVEL%

:usage
echo usage: run.bat ^<model^> [profile]
echo   qwen38-27b       tuned (default), int8, c8
echo   qwen36-35b-a3b   tuned (default)
exit /b 0
