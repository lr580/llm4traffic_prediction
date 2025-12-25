"""
LargeST EDA script for geodesic-distance analysis.

Given the metadata file shipped with LargeST (``data/LargeST/ca_meta.csv``),
this script computes the great-circle (haversine) distance between detector
pairs, reports descriptive statistics, and visualizes the distribution with a
histogram. When ``--edges-only`` is provided, only detector pairs that have
edges in the supplied adjacency matrix (default ``ca_rn_adj.npy``) are analyzed.
"""

''' python utils/baselines/eda/LargeST_EDA_geo.py --edges-only --bins 30 --dtype float32 --savefig data/LargeST/geo_hist_edges.png

Loaded 8600 detector coordinates from data\LargeST\ca_meta.csv
Using adjacency-constrained pairs: 121,759 unique undirected edges (threshold > 0.0).

=== Pairwise geodesic distance summary (kilometres) ===
Total pairs considered: 121,759
Minimum distance:       0.001 km
Maximum distance:       4.010 km
Mean +/- std:           2.248 +/- 1.118 km

Detailed counts for small distances:
  <=   1.00 km:     21,439 pairs ( 17.61%)
  <=   5.00 km:    121,759 pairs (100.00%)
  <=  10.00 km:    121,759 pairs (100.00%)
  <=  20.00 km:    121,759 pairs (100.00%)
  <=  50.00 km:    121,759 pairs (100.00%)
  <= 100.00 km:    121,759 pairs (100.00%)

Low-end percentiles (km):
  P 0.1:    0.015
  P 0.5:    0.021
  P 1.0:    0.025
  P 2.0:    0.041
  P 5.0:    0.320
  P10.0:    0.629

Selected percentiles (km):
   P01:    0.025
   P05:    0.320
   P25:    1.351
   P50:    2.353
   P75:    3.213
   P95:    3.847
   P99:    3.969

Histogram bins: 30, total counts: 121,759
First few bins (km range -> count):
  Bin 01: [   0.00,    0.13) km -> 3,630
  Bin 02: [   0.13,    0.27) km -> 1,599
  Bin 03: [   0.27,    0.40) km -> 2,258
  Bin 04: [   0.40,    0.53) km -> 2,652
  Bin 05: [   0.53,    0.67) km -> 3,059
Histogram saved to data\LargeST\geo_hist_edges.png '''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6_371.0088  # WGS84 mean Earth radius in kilometres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute geodesic distance statistics for LargeST detectors."
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/LargeST/ca_meta.csv"),
        help="Path to ca_meta.csv (default: data/LargeST/ca_meta.csv)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of equally spaced bins for the histogram (default: 40).",
    )
    parser.add_argument(
        "--edges-only",
        action="store_true",
        help="Restrict the computation to detector pairs that have edges in the adjacency matrix.",
    )
    parser.add_argument(
        "--adj",
        type=Path,
        default=Path("data/LargeST/ca_rn_adj.npy"),
        help="Adjacency matrix path used when --edges-only is enabled (default: data/LargeST/ca_rn_adj.npy).",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Minimum adjacency weight (exclusive) treated as an edge when --edges-only is enabled.",
    )
    parser.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="Optional path to save the histogram figure (PNG). Show interactively when omitted.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Precision used to store distances in memory (default: float32).",
    )
    parser.add_argument(
        "--small-thresholds",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        help="Distance thresholds (km) for detailed counts when --edges-only is enabled.",
    )
    return parser.parse_args()


def load_coordinates(meta_path: Path) -> np.ndarray:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    df = pd.read_csv(meta_path, usecols=["Lat", "Lng"])
    coords = np.deg2rad(df[["Lat", "Lng"]].to_numpy(dtype=np.float64))
    return coords


def haversine_distance(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """
    Vectorized haversine distance between two latitude/longitude arrays
    (broadcasting is supported). Lat/lon must be provided in radians.
    """
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_KM * c


def compute_pairwise_distances(
    coords: np.ndarray, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Compute haversine distances between every unordered pair of coordinates.

    Returns:
        distances: Flat array of length n*(n-1)/2 storing the upper triangle.
    """
    n = coords.shape[0]
    total_pairs = n * (n - 1) // 2
    if total_pairs <= 0:
        return np.empty(0, dtype=dtype)

    distances = np.empty(total_pairs, dtype=dtype)
    cursor = 0
    for idx in range(n - 1):
        lat1, lon1 = coords[idx]
        lat_rest = coords[idx + 1 :, 0]
        lon_rest = coords[idx + 1 :, 1]
        chunk = haversine_distance(lat1, lon1, lat_rest, lon_rest)
        chunk = chunk.astype(dtype, copy=False)
        size = chunk.size
        distances[cursor : cursor + size] = chunk
        cursor += size
    return distances


def compute_distances_for_pairs(
    coords: np.ndarray, pairs: np.ndarray, dtype: np.dtype = np.float32
) -> np.ndarray:
    if pairs.size == 0:
        return np.empty(0, dtype=dtype)
    lat_i = coords[pairs[:, 0], 0]
    lon_i = coords[pairs[:, 0], 1]
    lat_j = coords[pairs[:, 1], 0]
    lon_j = coords[pairs[:, 1], 1]
    dists = haversine_distance(lat_i, lon_i, lat_j, lon_j)
    return dists.astype(dtype, copy=False)


def load_adjacent_pairs(adj_path: Path, threshold: float) -> np.ndarray:
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adj_path}")
    if threshold < 0:
        raise ValueError("Edge threshold must be non-negative.")
    adj = np.load(adj_path, mmap_mode="r")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square; got shape {adj.shape}.")
    mask = np.array(adj > threshold, dtype=bool)
    np.fill_diagonal(mask, False)
    mask |= mask.T  # treat the graph as undirected for geodesic distance stats
    upper = np.triu(mask, k=1)
    i_idx, j_idx = np.where(upper)
    if i_idx.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([i_idx, j_idx], axis=1)


def summarize_distances(distances: np.ndarray) -> dict:
    if distances.size == 0:
        return {}
    percentiles = np.percentile(distances, [1, 5, 25, 50, 75, 95, 99])
    summary = {
        "count": int(distances.size),
        "min_km": float(distances.min()),
        "max_km": float(distances.max()),
        "mean_km": float(distances.mean()),
        "std_km": float(distances.std(ddof=0)),
        "percentiles_km": {
            "p01": float(percentiles[0]),
            "p05": float(percentiles[1]),
            "p25": float(percentiles[2]),
            "p50": float(percentiles[3]),
            "p75": float(percentiles[4]),
            "p95": float(percentiles[5]),
            "p99": float(percentiles[6]),
        },
    }
    return summary


def report_small_value_stats(distances: np.ndarray, thresholds: Iterable[float]) -> None:
    if distances.size == 0:
        print("No distances to report small-value statistics.")
        return
    cleaned_thresholds = sorted({thr for thr in thresholds if thr > 0})
    if not cleaned_thresholds:
        print("No positive thresholds supplied for small-value statistics.")
        return
    total = distances.size
    print("\nDetailed counts for small distances:")
    for thr in cleaned_thresholds:
        count = int((distances <= thr).sum())
        pct = count / total * 100.0
        print(f"  <= {thr:6.2f} km: {count:10,} pairs ({pct:6.2f}%)")

    fine_percentiles = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    percentile_values = np.percentile(distances, fine_percentiles)
    print("\nLow-end percentiles (km):")
    for p, value in zip(fine_percentiles, percentile_values):
        print(f"  P{p:4.1f}: {value:8.3f}")


def build_histogram(
    distances: np.ndarray, bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    if bins <= 0:
        raise ValueError("Number of bins must be positive.")
    if distances.size == 0:
        return np.array([]), np.array([])
    max_range = max(float(distances.max()), 1.0)
    bin_edges = np.linspace(0.0, max_range, bins + 1)
    counts, edges = np.histogram(distances, bins=bin_edges)
    return counts, edges


def plot_histogram(counts: np.ndarray, edges: np.ndarray, save_path: Path | None) -> None:
    if counts.size == 0:
        print("No distances computed; skipping histogram plot.")
        return
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    plt.figure(figsize=(12, 5))
    plt.bar(centers, counts, width=widths, align="center", edgecolor="black", alpha=0.8)
    plt.xlabel("Great-circle distance (km)")
    plt.ylabel("Detector pair count")
    plt.title("LargeST detector pairwise geodesic distance distribution")
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()
    coords = load_coordinates(args.meta)
    print(f"Loaded {coords.shape[0]} detector coordinates from {args.meta}")

    dtype = np.dtype(args.dtype)
    if args.edges_only:
        pairs = load_adjacent_pairs(args.adj, args.edge_threshold)
        print(
            f"Using adjacency-constrained pairs: {pairs.shape[0]:,} unique undirected edges "
            f"(threshold > {args.edge_threshold})."
        )
        distances = compute_distances_for_pairs(coords, pairs, dtype=dtype)
    else:
        distances = compute_pairwise_distances(coords, dtype=dtype)

    summary = summarize_distances(distances)
    if not summary:
        print("No pairs to analyze.")
        return

    print("\n=== Pairwise geodesic distance summary (kilometres) ===")
    print(f"Total pairs considered: {summary['count']:,}")
    print(f"Minimum distance:       {summary['min_km']:.3f} km")
    print(f"Maximum distance:       {summary['max_km']:.3f} km")
    print(f"Mean +/- std:           {summary['mean_km']:.3f} +/- {summary['std_km']:.3f} km")

    if args.edges_only:
        report_small_value_stats(distances, args.small_thresholds)

    print("\nSelected percentiles (km):")
    for label, value in summary["percentiles_km"].items():
        print(f"  {label.upper():>4}: {value:8.3f}")

    counts, edges = build_histogram(distances, args.bins)
    print(f"\nHistogram bins: {counts.size}, total counts: {counts.sum():,}")

    bin_info = zip(range(1, counts.size + 1), edges[:-1], edges[1:], counts)
    print("First few bins (km range -> count):")
    for idx, left, right, count in list(bin_info)[: min(5, counts.size)]:
        print(f"  Bin {idx:02d}: [{left:7.2f}, {right:7.2f}) km -> {count:,}")

    plot_histogram(counts, edges, args.savefig)


if __name__ == "__main__":
    main()
