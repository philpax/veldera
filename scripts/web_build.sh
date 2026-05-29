#!/bin/sh
set -e
cargo build --release -p veldera --target wasm32-unknown-unknown --no-default-features
wasm-bindgen \
    --no-typescript \
    --target web \
    --out-dir ./build/ \
    --out-name "veldera" \
    ./target/wasm32-unknown-unknown/release/veldera.wasm

# Optimize WASM if wasm-opt is available.
if command -v wasm-opt > /dev/null 2>&1; then
    wasm-opt -Oz -o build/veldera_bg.wasm build/veldera_bg.wasm
fi

# Copy runtime assets next to the page. Bevy fetches assets over HTTP from
# `./assets/` relative to index.html on the web, so without this the config
# TOML, models, audio, fonts, and topography all 404 in a production build.
rm -rf build/assets
cp -r client/veldera/assets build/assets

cat <<EOF > build/index.html
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>veldera</title>
  </head>
  <body style="margin: 0px; width: 100vw; height: 100vh;">
    <script type="module">
      import init from "./veldera.js";

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
