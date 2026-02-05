{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust toolchain
    rustup

    # Build dependencies
    pkg-config
    protobuf

    # Native dependencies for Bevy
    udev
    alsa-lib
    vulkan-loader

    # X11 dependencies
    xorg.libX11
    xorg.libXcursor
    xorg.libXi
    xorg.libXrandr

    # Wayland dependencies
    libxkbcommon
    wayland
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    udev
    alsa-lib
    vulkan-loader
    libxkbcommon
    wayland
    xorg.libX11
    xorg.libXcursor
    xorg.libXi
    xorg.libXrandr
  ]);

  shellHook = ''
    echo "Rust rocktree development environment"
    echo "Run 'cargo build' to build, 'cargo run' to run the client"
  '';
}
