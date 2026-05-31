#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
# ]
# ///
"""
Analyze vehicle telemetry CSV files to diagnose physics issues.

Usage:
    uv run analyze_telemetry.py <telemetry.csv> [--launches] [--summary] [--around <time>]

Examples:
    uv run analyze_telemetry.py telemetry.csv --summary
    uv run analyze_telemetry.py telemetry.csv --launches
    uv run analyze_telemetry.py telemetry.csv --around 8.5
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console encoding issues.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import polars as pl


def load_telemetry(path: Path) -> pl.DataFrame:
    """Load telemetry CSV file."""
    return pl.read_csv(path)


def find_launches(df: pl.DataFrame, v_vel_threshold: float = 10.0) -> pl.DataFrame:
    """Find moments where vertical velocity spikes upward (potential launches)."""
    return df.with_columns(
        pl.col("v_vel").diff().alias("v_vel_delta")
    ).filter(pl.col("v_vel_delta") > v_vel_threshold)


def find_high_velocity(df: pl.DataFrame, v_vel_threshold: float = 15.0) -> pl.DataFrame:
    """Find moments where vertical velocity is high (already launched)."""
    return df.filter(pl.col("v_vel") > v_vel_threshold)


def find_force_spikes(df: pl.DataFrame, weight: float, ratio_threshold: float = 3.0) -> pl.DataFrame:
    """Find moments where hover force exceeds threshold times weight."""
    return df.filter(pl.col("hover_mag") > weight * ratio_threshold)


def get_context(df: pl.DataFrame, time: float, window: float = 0.5) -> pl.DataFrame:
    """Get rows around a specific time."""
    return df.filter(
        (pl.col("t") >= time - window) & (pl.col("t") <= time + window)
    )


def print_summary(df: pl.DataFrame):
    """Print summary statistics."""
    print("=" * 60)
    print("TELEMETRY SUMMARY")
    print("=" * 60)

    t_min = df["t"].min()
    t_max = df["t"].max()
    mass = df["mass"].item(-1)
    weight = mass * 9.81

    print(f"\nTime range: {t_min:.2f}s - {t_max:.2f}s ({t_max - t_min:.2f}s total)")
    print(f"Samples: {len(df)}")
    print(f"Mass: {mass:.1f} kg")
    print(f"Weight: {weight:.0f} N")

    print("\n--- Altitude ---")
    valid_alt = df.filter(pl.col("altitude") > 0)
    if len(valid_alt) > 0:
        print(f"  Min (valid): {valid_alt['altitude'].min():.3f} m")
        print(f"  Max (valid): {valid_alt['altitude'].max():.3f} m")
        print(f"  Mean (valid): {valid_alt['altitude'].mean():.3f} m")
    no_raycast = df.filter(pl.col("altitude") < 0)
    print(f"  No raycast hit: {len(no_raycast)} samples ({len(no_raycast)/len(df)*100:.1f}%)")

    print("\n--- Velocity ---")
    print(f"  Max speed: {df['speed'].max():.1f} m/s ({df['speed'].max() * 3.6:.1f} km/h)")
    print(f"  Max v_vel (up): {df['v_vel'].max():.2f} m/s")
    print(f"  Min v_vel (down): {df['v_vel'].min():.2f} m/s")

    print("\n--- Forces ---")
    print(f"  Max hover force: {df['hover_mag'].max():.0f} N")
    print(f"  Max hover/weight: {df['hover_mag'].max() / weight:.2f}x")
    print(f"  Mean hover (grounded): {df.filter(pl.col('grounded') == 1)['hover_mag'].mean():.0f} N")

    print("\n--- Grounded ---")
    grounded_pct = df.filter(pl.col("grounded") == 1).height / len(df) * 100
    print(f"  Time grounded: {grounded_pct:.1f}%")

    # Detect problems
    print("\n--- PROBLEM DETECTION ---")

    # High vertical velocity
    high_vvel = find_high_velocity(df, 15.0)
    if len(high_vvel) > 0:
        print(f"  HIGH VERTICAL VELOCITY: {len(high_vvel)} samples with v_vel > 15 m/s")
        first = high_vvel.row(0, named=True)
        print(f"    First at t={first['t']:.3f}s: v_vel={first['v_vel']:.1f} m/s, alt={first['altitude']:.2f}m")

    # Force spikes
    force_spikes = find_force_spikes(df, weight, 5.0)
    if len(force_spikes) > 0:
        print(f"  FORCE SPIKES: {len(force_spikes)} samples with hover > 5x weight")
        first = force_spikes.row(0, named=True)
        print(f"    First at t={first['t']:.3f}s: force={first['hover_mag']:.0f}N ({first['hover_mag']/weight:.1f}x weight)")

    # Delta spikes
    launches = find_launches(df, 10.0)
    if len(launches) > 0:
        print(f"  VELOCITY SPIKES: {len(launches)} samples with v_vel delta > 10 m/s")


def print_around_time(df: pl.DataFrame, time: float, window: float = 0.5):
    """Print telemetry around a specific time."""
    context = get_context(df, time, window)

    if len(context) == 0:
        print(f"No data found around t={time}s")
        return

    print(f"Telemetry around t={time}s (±{window}s):\n")

    # Select key columns
    cols = ["t", "altitude", "v_vel", "grounded", "hover_mag", "pitch_deg", "speed"]
    available = [c for c in cols if c in context.columns]

    with pl.Config(tbl_rows=100, fmt_float="full"):
        print(context.select(available))


def print_problem_context(df: pl.DataFrame):
    """Print context around detected problems."""
    mass = df["mass"].item(-1)
    weight = mass * 9.81

    # Find first high velocity event
    high_vvel = find_high_velocity(df, 15.0)
    if len(high_vvel) > 0:
        first_time = high_vvel["t"].item(0)
        print(f"\n{'='*60}")
        print(f"CONTEXT: First high v_vel at t={first_time:.3f}s")
        print("=" * 60)

        # Get context starting a bit before
        context = get_context(df, first_time, window=0.3)
        cols = ["t", "altitude", "v_vel", "grounded", "hover_mag"]
        with pl.Config(tbl_rows=50):
            print(context.select(cols))

    # Find first force spike
    force_spikes = find_force_spikes(df, weight, 5.0)
    if len(force_spikes) > 0:
        first_time = force_spikes["t"].item(0)
        print(f"\n{'='*60}")
        print(f"CONTEXT: First force spike at t={first_time:.3f}s")
        print("=" * 60)

        context = get_context(df, first_time, window=0.3)
        cols = ["t", "altitude", "v_vel", "grounded", "hover_mag"]
        with pl.Config(tbl_rows=50):
            print(context.select(cols))


def main():
    parser = argparse.ArgumentParser(description="Analyze vehicle telemetry")
    parser.add_argument("file", type=Path, help="Telemetry CSV file")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")
    parser.add_argument("--context", action="store_true", help="Show context around problems")
    parser.add_argument("--around", type=float, help="Show telemetry around specific time")
    parser.add_argument("--window", type=float, default=0.5, help="Time window for --around")

    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    df = load_telemetry(args.file)

    # Default to summary + context if no specific action requested
    if not any([args.summary, args.context, args.around]):
        args.summary = True
        args.context = True

    if args.summary:
        print_summary(df)

    if args.context:
        print_problem_context(df)

    if args.around is not None:
        print_around_time(df, args.around, args.window)


if __name__ == "__main__":
    main()
