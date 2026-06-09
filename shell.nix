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
    libX11
    libXcursor
    libXi
    libXrandr

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
    libX11
    libXcursor
    libXi
    libXrandr
  ]);
}
