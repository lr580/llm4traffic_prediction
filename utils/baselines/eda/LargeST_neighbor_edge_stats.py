"""
LargeST 节点邻边测地距离、边权与度统计工具

主要功能：
1. 依托 ``ca_meta.csv`` 中的经纬度与 ``ca_rn_adj.npy`` 边权，计算每个节点全部邻边的测地距离；
2. 输出每个节点的距离统计量（最小/最大/均值/中位数/标准差/四分位）以及边权统计量；
3. 统计整体度分布、距离分布、边权分布信息，并给出直方图可视化；
4. 所有结果均默认写入 ``utils/baselines/eda`` 目录。

示例：
    python utils/baselines/eda/LargeST_neighbor_edge_stats.py \
        --edge-threshold 0.0 \
        --output-csv utils/baselines/eda/LargeST_neighbor_edge_stats.csv \
        --output-summary utils/baselines/eda/LargeST_neighbor_edge_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6_371.0088
METERS_PER_KM = 1_000.0
DEFAULT_FONT_FALLBACK = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["font.sans-serif"] = DEFAULT_FONT_FALLBACK
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class HistogramConfig:
    series: str
    title: str
    xlabel: str
    filename: str
    bins: int = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 LargeST 中每个节点的邻边测地距离、边权与度分布，并输出可视化结果。"
    )
    parser.add_argument("--meta", type=Path, default=Path("data/LargeST/ca_meta.csv"), help="节点经纬度文件路径")
    parser.add_argument("--adj", type=Path, default=Path("data/LargeST/ca_rn_adj.npy"), help="邻接矩阵 npy 文件路径")
    parser.add_argument(
        "--edge-threshold", type=float, default=0.0, help="只保留权重大于该阈值的边（默认 0 即保留全部）"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("utils/baselines/eda/LargeST_neighbor_edge_stats.csv"),
        help="每个节点的统计量 CSV 输出路径",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("utils/baselines/eda/LargeST_neighbor_edge_summary.json"),
        help="整体统计量 JSON 输出路径",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("utils/baselines/eda"),
        help="可视化图片输出目录，文件名由脚本自动生成",
    )
    parser.add_argument("--show", action="store_true", help="是否显示 matplotlib 窗口")
    return parser.parse_args()


def load_coordinates(meta_path: Path) -> np.ndarray:
    if not meta_path.exists():
        raise FileNotFoundError(f"未找到节点元数据: {meta_path}")
    df = pd.read_csv(meta_path, usecols=["Lat", "Lng"])
    coords = np.deg2rad(df[["Lat", "Lng"]].to_numpy(dtype=np.float64))
    return coords


def load_adjacency(adj_path: Path) -> np.ndarray:
    if not adj_path.exists():
        raise FileNotFoundError(f"未找到邻接矩阵: {adj_path}")
    adj = np.load(adj_path, mmap_mode="r")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"邻接矩阵必须为方阵，当前形状: {adj.shape}")
    return adj


def haversine_distance(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_KM * c * METERS_PER_KM


def describe_array(values: Iterable[float]) -> Dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: None for k in ("count", "mean", "std", "min", "p25", "median", "p75", "max")}
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def build_node_statistics(coords: np.ndarray, adj: np.ndarray, threshold: float) -> pd.DataFrame:
    num_nodes = coords.shape[0]
    if adj.shape[0] != num_nodes:
        raise ValueError(f"邻接矩阵节点数 {adj.shape[0]} 与经纬度 {num_nodes} 不一致")

    adj_full = np.asarray(adj, dtype=np.float64)
    edge_mask = adj_full > threshold
    # 认为边无向，两个方向只要有一条即认为存在
    edge_mask = np.logical_or(edge_mask, edge_mask.T)
    np.fill_diagonal(edge_mask, False)

    rows: List[Dict[str, float | int | None]] = []
    for node_idx in range(num_nodes):
        neighbor_idx = np.where(edge_mask[node_idx])[0]
        degree = neighbor_idx.size
        entry: Dict[str, float | int | None] = {"node_index": node_idx, "degree": int(degree)}

        if degree == 0:
            # 无邻居直接填充空统计
            for key in ("min", "max", "mean", "median", "std", "p25", "p75"):
                entry[f"distance_{key}_m"] = np.nan
                entry[f"weight_{key}"] = np.nan
            entry["distance_count"] = 0
            entry["weight_count"] = 0
            rows.append(entry)
            continue

        lat1, lon1 = coords[node_idx]
        lat2 = coords[neighbor_idx, 0]
        lon2 = coords[neighbor_idx, 1]
        distances_m = haversine_distance(lat1, lon1, lat2, lon2)

        row_weights = adj_full[node_idx, neighbor_idx]
        col_weights = adj_full[neighbor_idx, node_idx]
        weights = np.maximum(row_weights, col_weights)

        distance_stats = describe_array(distances_m)
        weight_stats = describe_array(weights)

        entry["distance_count"] = int(distance_stats["count"]) if distance_stats["count"] is not None else 0
        entry["weight_count"] = int(weight_stats["count"]) if weight_stats["count"] is not None else 0

        for key in ("min", "max", "mean", "median", "std", "p25", "p75"):
            entry[f"distance_{key}_m"] = distance_stats.get(key)
            entry[f"weight_{key}"] = weight_stats.get(key)

        rows.append(entry)

    df = pd.DataFrame(rows)
    return df


def build_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float | None]]:
    summary: Dict[str, Dict[str, float | None]] = {}
    summary["degree"] = describe_array(df["degree"])
    distance_columns = [
        "distance_min_m",
        "distance_max_m",
        "distance_mean_m",
        "distance_median_m",
        "distance_std_m",
        "distance_p25_m",
        "distance_p75_m",
    ]
    weight_columns = [
        "weight_min",
        "weight_max",
        "weight_mean",
        "weight_median",
        "weight_std",
        "weight_p25",
        "weight_p75",
    ]
    for col in distance_columns:
        summary[col] = describe_array(df[col].dropna().to_numpy())
    for col in weight_columns:
        summary[col] = describe_array(df[col].dropna().to_numpy())
    return summary


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)
    print(f"节点级统计已保存：{path}")


def save_summary(summary: Dict[str, Dict[str, float | None]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"整体统计摘要已保存：{path}")


def plot_histograms(df: pd.DataFrame, figure_dir: Path, show: bool) -> List[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        HistogramConfig(
            series="degree",
            title="LargeST 节点度分布",
            xlabel="度（邻边数量）",
            filename="LargeST_degree_hist.png",
            bins=80,
        ),
        HistogramConfig(
            series="distance_mean_m",
            title="节点邻边测地距离均值分布",
            xlabel="距离均值（米）",
            filename="LargeST_distance_mean_hist.png",
        ),
        HistogramConfig(
            series="weight_mean",
            title="节点邻边边权均值分布",
            xlabel="边权均值",
            filename="LargeST_weight_mean_hist.png",
        ),
    ]

    saved_paths: List[Path] = []
    for cfg in configs:
        data = df[cfg.series].dropna().to_numpy()
        if data.size == 0:
            print(f"{cfg.series} 没有可视化数据，跳过绘图。")
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=cfg.bins, color="#1f77b4", edgecolor="black", alpha=0.85)
        plt.title(cfg.title)
        plt.xlabel(cfg.xlabel)
        plt.ylabel("频数")
        plt.grid(alpha=0.2, linestyle="--")
        plt.tight_layout()
        out_path = figure_dir / cfg.filename
        plt.savefig(out_path, dpi=300)
        saved_paths.append(out_path)
        print(f"图像已保存：{out_path}")
        if show:
            plt.show()
        else:
            plt.close()
    return saved_paths


def main() -> None:
    args = parse_args()
    coords = load_coordinates(args.meta)
    adj = load_adjacency(args.adj)

    df = build_node_statistics(coords, adj, args.edge_threshold)
    save_dataframe(df, args.output_csv)

    summary = build_summary(df)
    summary["meta"] = {
        "node_count": int(df.shape[0]),
        "edge_threshold": args.edge_threshold,
        "source_meta": str(args.meta),
        "source_adj": str(args.adj),
    }
    save_summary(summary, args.output_summary)

    plot_histograms(df, args.figure_dir, args.show)


if __name__ == "__main__":
    main()
