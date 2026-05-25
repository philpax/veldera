#!/usr/bin/env bash
# Convert every character directory under source_assets/characters/mixamo
# into a single skinned glTF binary under client/veldera/assets/characters.
#
# Run from the repository root.
set -euo pipefail

cd "$(dirname "$0")/.."

src_root=source_assets/characters/mixamo
dst_root=client/veldera/assets/characters

if [[ ! -d "$src_root" ]]; then
  echo "no source directory at $src_root; nothing to convert"
  exit 0
fi

cargo build --release -p convert-character

for src_dir in "$src_root"/*/; do
  name=$(basename "$src_dir")
  out="$dst_root/$name.glb"
  echo "==> $name"
  ./target/release/convert-character "$src_dir" "$out"
done
