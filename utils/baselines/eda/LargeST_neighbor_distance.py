"""
LargeST 邻边测地距离统计工具。

功能概述：
1. 读取 ``ca_meta.csv`` 的经纬度信息与 ``ca_rn_adj.npy``（或自定义矩阵）的连边。
2. 对每个检测器节点，计算其所有邻边对应的真实测地距离（单位：米）及边权，按距离升序排序。
3. 依据各节点最小测地距离对整体结果排序，既可以输出完整 JSON，也能导出 CSV 摘要，并通过 CLI/图形化展示 Top-K 节点。

示例：
    python utils/baselines/eda/LargeST_neighbor_distance.py --output-json data/LargeST/neighbor_distances.json \
        --top-k 25 --savefig data/LargeST/neighbor_min_dist.svg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6_371.0088  # WGS84 平均地球半径（km）
METERS_PER_KM = 1_000.0
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 LargeST 每个节点的邻边测地距离（米），并输出排序结果。"
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/LargeST/ca_meta.csv"),
        help="元数据路径（默认: data/LargeST/ca_meta.csv）。",
    )
    parser.add_argument(
        "--adj",
        type=Path,
        default=Path("data/LargeST/ca_rn_adj.npy"),
        help="邻接矩阵路径（默认: data/LargeST/ca_rn_adj.npy）。",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="只保留权重大于该阈值的边（默认 0）。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/LargeST/neighbor_distance_stats.json"),
        help="完整 JSON 结果输出位置（默认: data/LargeST/neighbor_distance_stats.json）。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/LargeST/neighbor_min_summary.csv"),
        help="最小距离摘要 CSV 输出位置（默认: data/LargeST/neighbor_min_summary.csv）。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="CLI/图形展示的前 K 个节点（默认 20）。",
    )
    parser.add_argument(
        "--savefig",
        type=Path,
        default=None,
        help="若提供，则保存 Top-K 节点最小测地距离的柱状图至该路径（支持 SVG/PNG 等）。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示 matplotlib 图窗（默认仅在保存时静默绘制）。",
    )
    return parser.parse_args()


def load_coordinates(meta_path: Path) -> np.ndarray:
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到元数据文件: {meta_path}")
    df = pd.read_csv(meta_path, usecols=["Lat", "Lng"])
    coords = np.deg2rad(df[["Lat", "Lng"]].to_numpy(dtype=np.float64))
    return coords


def load_adjacency(adj_path: Path) -> np.ndarray:
    if not adj_path.exists():
        raise FileNotFoundError(f"找不到邻接矩阵: {adj_path}")
    adj = np.load(adj_path, mmap_mode="r")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"邻接矩阵必须为方阵，当前形状 {adj.shape}")
    return adj


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_KM * c  # 返回千米


def build_neighbor_stats(coords: np.ndarray, adj: np.ndarray, threshold: float) -> List[Dict]:
    """
    构建每个节点的邻居距离信息（单位：米）。

    返回 list，元素结构：
        {
            "node_index": int,
            "min_distance_m": float | None,
            "neighbors": [
                {"neighbor_index": int, "distance_m": float, "weight": float},
                ...
            ]
        }
    """
    num_nodes = coords.shape[0]
    stats: List[Dict] = []

    for i in range(num_nodes):
        row_mask = adj[i] > threshold
        col_mask = adj[:, i] > threshold
        neighbor_idx = np.where(row_mask | col_mask)[0]
        neighbor_idx = neighbor_idx[neighbor_idx != i]

        if neighbor_idx.size == 0:
            stats.append(
                {
                    "node_index": i,
                    "min_distance_m": None,
                    "neighbors": [],
                }
            )
            continue

        lat1 = coords[i, 0]
        lon1 = coords[i, 1]
        lat2 = coords[neighbor_idx, 0]
        lon2 = coords[neighbor_idx, 1]
        distances_km = haversine_distance(lat1, lon1, lat2, lon2)
        distances_m = distances_km * METERS_PER_KM

        row_weights = adj[i, neighbor_idx]
        col_weights = adj[neighbor_idx, i]
        weights = np.where(row_weights > threshold, row_weights, col_weights)

        order = np.argsort(distances_m)
        neighbor_list = []
        for idx in order:
            neighbor_list.append(
                {
                    "neighbor_index": int(neighbor_idx[idx]),
                    "distance_m": float(distances_m[idx]),
                    "weight": float(weights[idx]),
                }
            )

        stats.append(
            {
                "node_index": i,
                "min_distance_m": float(distances_m[order[0]]),
                "neighbors": neighbor_list,
            }
        )

    return stats


def sort_stats_by_min_distance(stats: List[Dict]) -> List[Dict]:
    def sort_key(entry: Dict):
        min_dist = entry.get("min_distance_m")
        if min_dist is None:
            return math.inf
        return min_dist

    return sorted(stats, key=sort_key)


def save_results(stats_sorted: List[Dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats_sorted, f, ensure_ascii=False, indent=2)
    print(f"完整 JSON 结果已保存至 {json_path}")

    summary_rows = []
    for entry in stats_sorted:
        summary_rows.append(
            {
                "node_index": entry["node_index"],
                "min_distance_m": entry.get("min_distance_m"),
                "neighbor_count": len(entry["neighbors"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)
    print(f"最小距离摘要已保存至 {csv_path}")


def print_cli_summary(stats_sorted: List[Dict], top_k: int) -> None:
    print("\n=== 最小测地距离 Top 列表（单位：米） ===")
    shown = 0
    for entry in stats_sorted:
        min_dist = entry.get("min_distance_m")
        if min_dist is None:
            continue
        print(
            f"节点 {entry['node_index']:4d} | 最小距离 {min_dist:.1f} m | 邻居数 {len(entry['neighbors'])}"
        )
        for neighbor in entry["neighbors"][:5]:
            print(
                f"    -> 邻居 {neighbor['neighbor_index']:4d} | "
                f"距离 {neighbor['distance_m']:.1f} m | 边权 {neighbor['weight']:.4f}"
            )
        shown += 1
        if shown >= top_k:
            break
    if shown == 0:
        print("没有发现任何包含邻居的节点。")


def plot_top_min_distances(stats_sorted: List[Dict], top_k: int, savefig: Path | None, show: bool) -> None:
    filtered = [entry for entry in stats_sorted if entry.get("min_distance_m") is not None]
    if not filtered:
        print("没有可视化的数据（所有节点都没有邻居）。")
        return
    top_entries = filtered[: top_k if top_k > 0 else len(filtered)]
    node_ids = [entry["node_index"] for entry in top_entries]
    min_dists = [entry["min_distance_m"] for entry in top_entries]

    plt.figure(figsize=(max(8, top_k * 0.4), 6))
    bars = plt.bar(range(len(node_ids)), min_dists, color="steelblue")
    plt.xticks(range(len(node_ids)), node_ids, rotation=45)
    plt.ylabel("最小测地距离 (m)")
    plt.xlabel("节点索引")
    plt.title(f"最小测地距离 Top {len(node_ids)} 节点")
    for bar, dist in zip(bars, min_dists):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{dist:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()

    if savefig:
        savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savefig, dpi=300)
        print(f"柱状图已保存至 {savefig}")
    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    coords = load_coordinates(args.meta)
    adj = load_adjacency(args.adj)
    if adj.shape[0] != coords.shape[0]:
        raise ValueError(
            f"邻接矩阵节点数 {adj.shape[0]} 与元数据节点数 {coords.shape[0]} 不一致。"
        )

    stats = build_neighbor_stats(coords, adj, args.edge_threshold)
    stats_sorted = sort_stats_by_min_distance(stats)

    save_results(stats_sorted, args.output_json, args.output_csv)
    print_cli_summary(stats_sorted, args.top_k)
    plot_top_min_distances(stats_sorted, args.top_k, args.savefig, args.show)


if __name__ == "__main__":
    main()
