#!/bin/sh
set -e
cargo build --release -p veldera-viewer --target wasm32-unknown-unknown --no-default-features
wasm-bindgen \
    --no-typescript \
    --target web \
    --out-dir ./build/ \
    --out-name "veldera_viewer" \
    ./target/wasm32-unknown-unknown/release/veldera-viewer.wasm

# Optimize WASM if wasm-opt is available.
if command -v wasm-opt > /dev/null 2>&1; then
    wasm-opt -Oz -o build/veldera_viewer_bg.wasm build/veldera_viewer_bg.wasm
fi

cat <<EOF > build/index.html
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>veldera-viewer</title>
  </head>
  <body style="margin: 0px; width: 100vw; height: 100vh;">
    <script type="module">
      import init from "./veldera_viewer.js";

      init().catch((error) => {
        if (
          !error.message.startsWith(
            "Using exceptions for control flow, don't mind me. This isn't actually an error!"
          )
        ) {
          throw error;
        }
      });
    </script>
  </body>
</html>
EOF
