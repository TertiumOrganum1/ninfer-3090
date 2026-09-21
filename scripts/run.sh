#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Serve a model on one RTX 3090.
#
#   run.sh <model> [profile]
#
#   model             profiles
#   qwen38-27b        tuned (default), int8, c8
#   qwen36-35b-a3b    tuned (default)
#
# `tuned` is the recommended profile: rk8v4 KV, speculation plus the draft head, the memory flags,
# vision in overlay residency, and the tuned context cache with automatic prefix grid. `int8` and
# `c8` are the older reference profiles for the 27B -- one user at 64K of INT8 KV (the quality
# default), and eight lanes at 8K -- with every serving flag fixed.
#
# Every measurement behind these defaults, the memory model, and the reasoning for each flag are
# in docs/maintainer/launcher-profiles.md. What follows is what you need to run it.
#
# QWEN3.8-27B, `tuned`: two flag sets, each measured (docs/performance.md, "Recommended
# configurations"), chosen with NINFER_SPEC. The default is the fast one.
#
#   NINFER_SPEC=dflash2 (default): fastest at one stream, for context up to about 130K
#
#     --spec dflash2 --draft-tokens 7 --lm-head-draft \
#     --prefill-cublas --prefill-chunk 4096 \
#     --kv-dtype rk8v4 --embedding-q4 --gdn-state-fp16 \
#     --vision --vision-residency overlay
#
#   NINFER_SPEC=mtp: the longest context, still fast -- 200,000 tokens verified by loading it
#
#     --spec mtp --draft-tokens 3 --lm-head-draft \
#     --prefill-cublas --prefill-chunk 2048 \
#     --kv-dtype rk8v4 --embedding-q4 --lm-head-q6 --gdn-state-fp16 \
#     --vision --vision-residency overlay
#
# Against the previous defaults the DFlash2 set is about 1.7x on prefill and 1.39x on decode, for
# +0.156% perplexity from the cuBLAS route and +0.083% from rk8v4. It gives up context because
# DFlash2's draft weights and its refusal of --lm-head-q6 cost about 65K tokens between them: the
# same flags on DFlash2 load at 130K and fail at 150K. `none` is the mtp set without speculation.
# The qwen3_8_27b.ninfer that download-model.sh fetches is the DFlash2 bundle and carries the MTP
# weights too, so one file serves both.
#
# OVERRIDES, from the environment. All profiles: NINFER_MODEL (artifact path), NINFER_MODEL_DIR,
# NINFER_SERVER, NINFER_HOST, NINFER_PORT. `tuned` also: NINFER_CONTEXT, NINFER_CONCURRENCY,
# NINFER_KV_CAPACITY, NINFER_KV_DTYPE, NINFER_SPEC, NINFER_DRAFT_TOKENS, NINFER_PREFILL_CHUNK,
# NINFER_VISION (on|off), NINFER_VISION_RESIDENCY. Each spec's defaults (context, lanes, chunk)
# are the ones that fit; the context figures below are extrapolated for a headless card, so treat
# the first start as the confirmation and drop a rung if it refuses:
# 229376 / 212992 / 196608 / 163840 / 131072 / 114688 / 98304 / 65536.
# ------------------------------------------------------------------------------------------------
set -euo pipefail

usage() {
  printf 'usage: %s <model> [profile]\n' "${0##*/}"
  printf '  qwen38-27b       tuned (default), int8, c8\n'
  printf '  qwen36-35b-a3b   tuned (default)\n'
}

model_key="${1:-}"
profile="${2:-tuned}"
case "$model_key" in
  -h|--help) usage; exit 0 ;;
  qwen38-27b)     artifact='qwen3_8_27b.ninfer';     title='Qwen3.8-27B' ;;
  qwen36-35b-a3b) artifact='qwen3_6_35b_a3b.ninfer'; title='Qwen3.6-35B-A3B' ;;
  '') usage >&2; exit 2 ;;
  *) printf 'Unknown model: %s\n' "$model_key" >&2; usage >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/.." && pwd)"
# Two layouts reach this script: a checkout, where the artifacts sit in the repository's own
# models/ directory beside build-linux/, and an unpacked release archive, where the launcher
# sits next to ninfer-serve and models/. Probe for the checkout first, then the archive -- the
# same order the server lookup below uses.
#
# An explicit NINFER_MODEL_DIR is taken verbatim and never probed. Falling back past a
# directory the caller named turns their typo into a 'Missing model' error about a path they
# never mentioned, which is worse than failing on the one they did.
if [[ -n "${NINFER_MODEL_DIR:-}" ]]; then
  model_dir="$NINFER_MODEL_DIR"
else
  # Test for the artifact, not merely for a models/ directory. An archive unpacked below a
  # directory that happens to have its own models/ would otherwise stop at the parent and
  # never look beside the launcher, failing while its own artifact sits right there.
  model_dir="$root/models"
  [[ -f "$model_dir/$artifact" ]] || model_dir="$script_dir/models"
fi
MODEL="${NINFER_MODEL:-$model_dir/$artifact}"
HOST="${NINFER_HOST:-127.0.0.1}"
PORT="${NINFER_PORT:-8080}"

server="${NINFER_SERVER:-$root/build-linux/apps/ninfer-serve}"
# Same two layouts as the artifact lookup above: build tree in a checkout, then the archive
# root, where this launcher sits beside the binary.
[[ -x "$server" ]] || server="$script_dir/ninfer-serve"

# The profile fixes the whole serving shape. `label` is the banner; `profile_args` is everything
# after --host/--port.
label=''
profile_args=()
case "$model_key/$profile" in
  qwen38-27b/tuned)
    # The speculative backend fixes everything that has to move with it: the prefill chunk (the
    # cuBLAS route's workspace scales with it), the context and lanes that fit, and --lm-head-q6,
    # which DFlash and DFlash2 refuse.
    SPEC="${NINFER_SPEC:-dflash2}"
    case "$SPEC" in
      dflash2)
        spec_args=(--spec dflash2 --draft-tokens "${NINFER_DRAFT_TOKENS:-7}" --lm-head-draft)
        memory_args=()
        default_context=131072; default_concurrency=1; default_chunk=4096
        spec_label="DFlash2 K=${NINFER_DRAFT_TOKENS:-7} + draft head" ;;
      mtp)
        spec_args=(--spec mtp --draft-tokens "${NINFER_DRAFT_TOKENS:-3}" --lm-head-draft)
        memory_args=(--lm-head-q6)
        default_context=212992; default_concurrency=2; default_chunk=2048
        spec_label="MTP${NINFER_DRAFT_TOKENS:-3} + draft head, Q6 head (longest context)" ;;
      none)
        spec_args=()
        memory_args=()
        default_context=212992; default_concurrency=2; default_chunk=2048
        spec_label='no speculation' ;;
      *) printf 'NINFER_SPEC must be dflash2, mtp or none, got %s\n' "$SPEC" >&2; exit 2 ;;
    esac
    CONTEXT="${NINFER_CONTEXT:-$default_context}"
    CONCURRENCY="${NINFER_CONCURRENCY:-$default_concurrency}"
    KV_CAPACITY="${NINFER_KV_CAPACITY:-$CONTEXT}"
    KV_DTYPE="${NINFER_KV_DTYPE:-rk8v4}"
    PREFILL_CHUNK="${NINFER_PREFILL_CHUNK:-$default_chunk}"
    profile_args=(
      --max-concurrency "$CONCURRENCY" --max-context "$CONTEXT" --kv-capacity "$KV_CAPACITY"
      --kv-dtype "$KV_DTYPE"
      ${spec_args[@]+"${spec_args[@]}"}
      --embedding-q4 ${memory_args[@]+"${memory_args[@]}"} --gdn-state-fp16
      --prefill-cublas --prefill-chunk "$PREFILL_CHUNK"
    )
    label="C$CONCURRENCY  |  context $CONTEXT  |  KV pool $KV_CAPACITY  |  $KV_DTYPE  |  $spec_label"
    prefill_note="Prefill: cuBLAS route, chunk $PREFILL_CHUNK"
    if [[ "$SPEC" == 'dflash2' ]]; then
      hint="Need more than ~130K context?  NINFER_SPEC=mtp ${0##*/} $model_key  (longer context, slower decode)"
    fi ;;

  qwen36-35b-a3b/tuned)
    SPEC="${NINFER_SPEC:-mtp}"
    case "$SPEC" in
      mtp)  spec_args=(--spec mtp --draft-tokens "${NINFER_DRAFT_TOKENS:-3}" --lm-head-draft --mtp-experts-q4)
            spec_label="MTP${NINFER_DRAFT_TOKENS:-3} + draft head" ;;
      none) spec_args=(); spec_label='no speculation' ;;
      *) printf 'NINFER_SPEC must be mtp or none, got %s\n' "$SPEC" >&2; exit 2 ;;
    esac
    CONTEXT="${NINFER_CONTEXT:-262144}"
    CONCURRENCY="${NINFER_CONCURRENCY:-2}"
    KV_CAPACITY="${NINFER_KV_CAPACITY:-$CONTEXT}"
    KV_DTYPE="${NINFER_KV_DTYPE:-rk8v4}"
    PREFILL_CHUNK="${NINFER_PREFILL_CHUNK:-512}"
    profile_args=(
      --max-concurrency "$CONCURRENCY" --max-context "$CONTEXT" --kv-capacity "$KV_CAPACITY"
      --kv-dtype "$KV_DTYPE"
      ${spec_args[@]+"${spec_args[@]}"}
      --gdn-state-fp16 --prefill-chunk "$PREFILL_CHUNK"
    )
    label="C$CONCURRENCY  |  context $CONTEXT  |  KV pool $KV_CAPACITY  |  $KV_DTYPE  |  $spec_label" ;;

  qwen38-27b/int8)
    profile_args=(
      --max-context 65536 --kv-capacity 65536
      --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000
      --prefill-chunk 1024 --kv-dtype int8
      --spec mtp --draft-tokens 3 --lm-head-draft
    )
    label='one request  |  64K context  |  INT8 KV  |  MTP3, ReplaySSM' ;;

  qwen38-27b/c8)
    profile_args=(
      --max-context 8192 --kv-capacity 16384
      --max-concurrency 8 --max-pending-requests 32 --pending-timeout-ms 600000
      --prefill-chunk 512 --kv-dtype int8
      --spec mtp --draft-tokens 3 --lm-head-draft
    )
    label='up to eight requests  |  8K context  |  INT8 KV  |  MTP3, ReplaySSM' ;;

  *)
    printf 'Model %s has no profile %s\n' "$model_key" "$profile" >&2
    usage >&2
    exit 2 ;;
esac

# Only `tuned` carries the context cache and vision: the reference profiles are deliberately
# minimal. Vision stays on there -- overlay residency keeps the tower host-pinned and streams each
# image through a borrowed device window, so it costs about 10 MiB of runtime reservation.
if [[ "$profile" == 'tuned' ]]; then
  VISION="${NINFER_VISION:-on}"
  case "$VISION" in
    on)  vision_args=(--vision --vision-residency "${NINFER_VISION_RESIDENCY:-overlay}")
         vision_label='vision (overlay)' ;;
    off) vision_args=(); vision_label='text only' ;;
    *) printf 'NINFER_VISION must be on or off, got %s\n' "$VISION" >&2; exit 2 ;;
  esac
  label="$label  |  $vision_label"
  profile_args+=(
    --max-pending-requests 16 --pending-timeout-ms 600000
    ${vision_args[@]+"${vision_args[@]}"}
    --max-private-continuations 8 --max-shared-prefixes 8 --host-state-slots 32 --host-kv-mib 8192
    --auto-prefix-grid
  )
fi

if [[ ! -x "$server" ]]; then
  printf 'Missing ninfer-serve (looked for %s)\n' "$server" >&2
  printf 'Build it first:  ./scripts/build.sh\n' >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  printf 'Missing model: %s\n' "$MODEL" >&2
  printf 'Download it first:  ./download-model.sh %s\n' "$model_key" >&2
  exit 1
fi

printf '%s  |  %s\n' "$title" "$label"
[[ -z "${prefill_note:-}" ]] || printf '%s\n' "$prefill_note"
if [[ "$profile" == 'tuned' ]]; then
  printf 'Cache: 8 shared / 8 private / 32 host states  |  automatic prefix grid on\n'
fi
[[ -z "${hint:-}" ]] || printf '%s\n' "$hint"
printf 'API: http://%s:%s/v1\n\n' "$HOST" "$PORT"

# --host-kv-mib 8192 is honoured in full here: on Linux this really does pin 8 GiB of host RAM, and
# it is host RAM, not device memory. run.bat passes the same number and gets far less -- WDDM
# charges a pinned host allocation against the card, so the runtime clamps to
# (free VRAM - 1 GiB) / 2. See docs/maintainer/launcher-profiles.md.
exec "$server" "$MODEL" --host "$HOST" --port "$PORT" "${profile_args[@]}"
