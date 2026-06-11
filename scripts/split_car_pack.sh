#!/usr/bin/env bash
# Split the bundled passenger-car-pack glb under source_assets into per-car
# glbs under client/veldera/assets/game/vehicles.
#
# Run from the repository root.
set -euo pipefail

cd "$(dirname "$0")/.."

src=source_assets/generic_passenger_car_pack.glb
dst=client/veldera/assets/game/vehicles

if [[ ! -f "$src" ]]; then
  echo "no source glb at $src; nothing to split"
  exit 0
fi

cargo build --release -p split-car-pack
./target/release/split-car-pack "$src" "$dst" "$@"
