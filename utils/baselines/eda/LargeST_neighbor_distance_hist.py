"""
LargeST 节点最近测地距离直方图。

读取 ``neighbor_distance_stats.json``（由 LargeST_neighbor_distance.py 生成），
提取每个节点的最小测地距离（单位：米），输出统计摘要，并绘制直方图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 neighbor_distance_stats.json 绘制最近测地距离直方图。"
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("data/LargeST/neighbor_distance_stats.json"),
        help="LargeST_neighbor_distance.py 生成的 JSON 文件路径。",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="直方图分箱数量（默认 60）。",
    )
    parser.add_argument(
        "--savefig",
        type=Path,
        default=Path("data/LargeST/neighbor_min_hist.svg"),
        help="直方图保存路径（默认 data/LargeST/neighbor_min_hist.svg）。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示图形窗口（默认不显示，仅保存）。",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        help="是否对横坐标使用对数尺度，适合长尾分布。",
    )
    return parser.parse_args()


def load_min_distances(stats_path: Path) -> np.ndarray:
    if not stats_path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as f:
        data: List[dict] = json.load(f)
    distances = [
        entry["min_distance_m"]
        for entry in data
        if entry.get("min_distance_m") is not None
    ]
    if not distances:
        raise ValueError("JSON 中没有任何包含邻居的节点，无法绘制直方图。")
    return np.array(distances, dtype=np.float64)


def print_summary(distances: np.ndarray) -> None:
    print("=== 最近测地距离统计（单位：米） ===")
    print(f"节点数量: {distances.size}")
    print(f"最小值: {distances.min():.3f}")
    print(f"最大值: {distances.max():.3f}")
    print(f"平均值: {distances.mean():.3f}")
    print(f"中位数: {np.median(distances):.3f}")
    for q in [5, 25, 50, 75, 95, 99]:
        print(f"P{q:02d}: {np.percentile(distances, q):.3f}")


def plot_hist(distances: np.ndarray, args: argparse.Namespace) -> None:
    plt.figure(figsize=(10, 6))
    if args.logx:
        # 为避免 log(0)，设置一个很小的正数作为最小值
        positive = distances[distances > 0]
        if positive.size == 0:
            raise ValueError("所有最小距离都为 0，无法使用 logx。")
        bins = np.logspace(
            np.log10(positive.min()),
            np.log10(positive.max()),
            args.bins + 1,
        )
        plt.hist(positive, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
        plt.xscale("log")
    else:
        plt.hist(distances, bins=args.bins, color="steelblue", edgecolor="black", alpha=0.8)
    plt.xlabel("最近测地距离 (m)")
    plt.ylabel("节点数量")
    plt.title("LargeST 节点最近测地距离直方图")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if args.savefig:
        args.savefig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.savefig, dpi=300)
        print(f"直方图已保存至 {args.savefig}")
    if args.show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    distances = load_min_distances(args.stats)
    print_summary(distances)
    plot_hist(distances, args)


if __name__ == "__main__":
    main()
