#!/usr/bin/env bash
set -euo pipefail

# The official v3 artifact revision. Its artifact-manifest.json lists the text, vision, mtp and
# dflash components; the DFlash bundle (from z-lab/Qwen3.6-35B-A3B-DFlash) is what --spec dflash
# needs. If you repin this, check the manifest still lists a dflash component, and update the size
# and checksum below from it.
revision='ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6'
expected_size=22790484480
expected_sha256='3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84'

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_6_35b_a3b.ninfer"

# curl -C - resumes by appending at the current file length, without checking what wrote those
# bytes. Because the pinned revision changed, a partial download of the *previous* artifact sits at
# exactly this path on any machine that ran the older script, and resuming onto it would splice the
# tail of one artifact onto the head of another: a file of entirely plausible size that is corrupt
# throughout. Staging under a name that carries the revision means a resume can only ever continue
# the same artifact, and the checks below are what promote it to the final name.
part="$model.$revision.part"

file_size() { wc -c < "$1" | tr -d '[:space:]'; }

# Verifies a file against expected_size and, unless NINFER_SKIP_SHA256=1, expected_sha256.
# Used both for an existing "$model" (so a same-sized-but-corrupt file is not accepted forever
# just because it happened to pass once, or was replaced out from under this script) and for a
# freshly downloaded "$part" -- one check that cannot drift out of sync with itself. A missing
# sha256sum/shasum fails closed rather than silently promoting an unverified file: the whole
# reason this artifact is checksummed is to catch a same-sized-but-corrupt file, and silently
# skipping that would defeat it, not just once, but for every future run on that host.
verify() {
  [ "$(file_size "$1")" = "$expected_size" ] || return 1
  [ "${NINFER_SKIP_SHA256:-0}" = '1' ] && return 0
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum -- "$1" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 -- "$1" | cut -d' ' -f1)"
  else
    printf 'No sha256sum or shasum found; cannot verify %s. Set NINFER_SKIP_SHA256=1 to accept it unverified.\n' "$1" >&2
    return 1
  fi
  [ "$actual" = "$expected_sha256" ]
}

mkdir -p -- "$model_dir"

if [ -f "$model" ]; then
  if verify "$model"; then
    printf 'Model already present: %s\n' "$model"
    exit 0
  fi
  printf '%s\n' "Existing $model did not verify against revision $revision; fetching the pinned one." >&2
  # A v2 file from an earlier release does not load any more, but it does not need to be fetched
  # again either: tools/upgrade_ninfer_v2_to_v3.py rewrites it locally in a couple of minutes.
  printf '%s\n' "If it is the v2 file from an earlier release, you can upgrade it instead of downloading:" \
    "  python tools/upgrade_ninfer_v2_to_v3.py $model $model.v3 && mv $model.v3 $model" >&2
fi

printf '%s\n' 'Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model (21.2 GiB)...'
if ! curl -L -C - --fail --output "$part" \
  "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/$revision/qwen3_6_35b_a3b.ninfer"; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi

if ! verify "$part"; then
  printf 'Downloaded file at %s failed verification against revision %s. Delete it and run this script again.\n' \
    "$part" "$revision" >&2
  exit 1
fi

mv -f -- "$part" "$model"
printf 'Model ready: %s\n' "$model"
