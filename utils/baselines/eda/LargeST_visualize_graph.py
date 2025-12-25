"""
LargeST 交通网络可视化。

脚本读取 ``ca_meta.csv`` 获取每个检测器的经纬度，再读取 ``ca_rn_adj.npy`` 中
大于指定阈值的边权，将所有点和边投影到经纬度平面。借助 matplotlib 的交互式
工具栏即可自由缩放和平移查看细节。可通过 ``--node-index`` 仅绘制某个结点及其
邻居，输入 ``-1`` 时会随机挑选一个检测器；还可以为每条边标注真实测地距离。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6_371.0088  # WGS84 平均地球半径（千米）
METERS_PER_KM = 1_000.0
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在经纬度平面可视化 LargeST 检测器及其邻接边。"
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/LargeST/ca_meta.csv"),
        help="LargeST 元数据文件路径（默认: data/LargeST/ca_meta.csv）。",
    )
    parser.add_argument(
        "--adj",
        type=Path,
        default=Path("data/LargeST/ca_rn_adj.npy"),
        help="邻接矩阵 npy 路径（默认: data/LargeST/ca_rn_adj.npy）。",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="只显示权重大于该阈值的边（默认 0）。",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=0,
        help="最多绘制的无向边数量，若超出则随机下采样（默认 15000，<=0 表示不限制）。",
    )
    parser.add_argument(
        "--node-index",
        type=int,
        default=None,
        help="仅绘制指定索引的结点及其邻点；输入 -1 表示随机挑选一个结点。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机采样（含随机选点）使用的随机种子。",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 9.0),
        metavar=("WIDTH", "HEIGHT"),
        help="图像尺寸（英寸）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="若提供，则保存为文件而非弹出窗口。",
    )
    parser.add_argument(
        "--node-size",
        type=float,
        default=4.0,
        help="结点散点的尺寸（默认 4）。",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=0.4,
        help="边的基础线宽，会根据权重自动缩放。",
    )
    parser.add_argument(
        "--annotate-limit",
        type=int,
        default=400,
        help="若边数量不超过该值，则在图中标注真实测地距离；<=0 表示始终标注。",
    )
    return parser.parse_args()


def load_coordinates(meta_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取元数据文件，返回纬度和经度数组。
    """
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到元数据文件: {meta_path}")
    df = pd.read_csv(meta_path, usecols=["Lat", "Lng"])
    lat = df["Lat"].to_numpy(dtype=np.float64)
    lon = df["Lng"].to_numpy(dtype=np.float64)
    return lat, lon


def load_edges(adj_path: Path, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取邻接矩阵，返回满足阈值条件的无向边端点索引及其权重。
    """
    if threshold < 0:
        raise ValueError("边阈值必须为非负数。")
    if not adj_path.exists():
        raise FileNotFoundError(f"找不到邻接矩阵: {adj_path}")
    adj = np.load(adj_path, mmap_mode="r")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"邻接矩阵必须是方阵，当前形状为 {adj.shape}。")
    mask = adj > threshold
    np.fill_diagonal(mask, False)
    undirected_mask = np.triu(np.logical_or(mask, mask.T), k=1)
    i_idx, j_idx = np.where(undirected_mask)
    weights = adj[i_idx, j_idx]
    return i_idx, j_idx, weights


def sample_edges(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    weights: np.ndarray,
    max_edges: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将边随机下采样到不超过 max_edges。
    """
    total = i_idx.size
    if max_edges <= 0 or total <= max_edges:
        return i_idx, j_idx, weights
    choice = rng.choice(total, size=max_edges, replace=False)
    return i_idx[choice], j_idx[choice], weights[choice]


def focus_on_node(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    weights: np.ndarray,
    node_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    仅保留包含指定结点的边，并输出需要绘制的结点索引集合。
    """
    mask = (i_idx == node_index) | (j_idx == node_index)
    i_sel = i_idx[mask]
    j_sel = j_idx[mask]
    w_sel = weights[mask]
    if w_sel.size == 0:
        visible_nodes = np.array([node_index], dtype=np.int64)
    else:
        visible_nodes = np.unique(np.concatenate((i_sel, j_sel, [node_index])))
    return i_sel, j_sel, w_sel, visible_nodes


def build_line_collection(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    weights: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    base_width: float,
) -> Tuple[LineCollection, Normalize]:
    """
    将边列表转换为 LineCollection 以便一次性添加到坐标轴。
    """
    segments = [[(lon[i], lat[i]), (lon[j], lat[j])] for i, j in zip(i_idx, j_idx)]
    norm = Normalize(vmin=float(weights.min()), vmax=float(weights.max()))
    line_widths = base_width * (0.5 + norm(weights))
    lc = LineCollection(
        segments,
        linewidths=line_widths,
        cmap="viridis",
        norm=norm,
        alpha=0.8,
    )
    lc.set_array(weights)
    return lc, norm


def compute_edge_distances_m(
    lat: np.ndarray, lon: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray
) -> np.ndarray:
    """
    根据经纬度计算边的测地距离（米）。
    """
    if i_idx.size == 0:
        return np.empty(0, dtype=np.float64)
    lat1 = np.deg2rad(lat[i_idx])
    lon1 = np.deg2rad(lon[i_idx])
    lat2 = np.deg2rad(lat[j_idx])
    lon2 = np.deg2rad(lon[j_idx])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_KM * c * METERS_PER_KM


def annotate_edges(
    ax: plt.Axes,
    distances_m: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
) -> None:
    """
    在每条边的中点处标注测地距离。
    """
    if distances_m.size == 0:
        return
    mid_lon = (lon[i_idx] + lon[j_idx]) / 2.0
    mid_lat = (lat[i_idx] + lat[j_idx]) / 2.0
    for x, y, dist in zip(mid_lon, mid_lat, distances_m):
        ax.text(
            x,
            y,
            f"{dist:.1f} m",
            fontsize=7,
            color="blue",
            alpha=0.7,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", pad=0.5),
        )


def plot_graph(
    lat: np.ndarray,
    lon: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
    visible_nodes: np.ndarray | None = None,
    focus_node: int | None = None,
) -> None:
    """
    绘制散点与边，可选地只显示部分结点并高亮中心结点。
    """
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    if visible_nodes is None:
        lon_to_plot = lon
        lat_to_plot = lat
    else:
        lon_to_plot = lon[visible_nodes]
        lat_to_plot = lat[visible_nodes]

    ax.scatter(
        lon_to_plot,
        lat_to_plot,
        s=args.node_size,
        c="black",
        alpha=0.75,
        linewidths=0,
        label="检测器",
    )

    if focus_node is not None:
        ax.scatter(
            lon[focus_node],
            lat[focus_node],
            s=args.node_size * 3,
            c="red",
            alpha=0.9,
            linewidths=0,
            label="中心结点",
            zorder=5,
        )

    if weights.size > 0:
        lc, _ = build_line_collection(
            i_idx, j_idx, weights, lat, lon, args.line_width
        )
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("边权")
        ax.legend(loc="upper right")
        print(
            f"已绘制 {weights.size} 条边 "
            f"(最小权重 {weights.min():.4f}, 最大权重 {weights.max():.4f})。"
        )
        distances_m = compute_edge_distances_m(lat, lon, i_idx, j_idx)
        should_annotate = (
            args.annotate_limit <= 0 or distances_m.size <= args.annotate_limit
        )
        if should_annotate:
            annotate_edges(ax, distances_m, lat, lon, i_idx, j_idx)
            print("已对所有边添加测地距离标注。")
        else:
            print(
                f"边数量 {distances_m.size:,} 超过 annotate-limit={args.annotate_limit}，"
                "为避免遮挡不添加标注，可调大该参数或过滤边。"
            )
    else:
        print("阈值限制后没有可绘制的边，仅展示结点。")

    ax.set_xlabel("经度 (°)")
    ax.set_ylabel("纬度 (°)")
    ax.set_title("LargeST 检测器图（使用工具栏可自由缩放）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300)
        print(f"图像已保存至 {args.output}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    lat, lon = load_coordinates(args.meta)
    num_nodes = lat.size
    print(f"已载入 {num_nodes} 个检测器坐标（来自 {args.meta}）。")

    i_idx, j_idx, weights = load_edges(args.adj, args.edge_threshold)
    print(
        f"权重大于 {args.edge_threshold} 的无向边共有 {weights.size:,} 条。"
    )

    visible_nodes = None
    focus_node = None

    if args.node_index is not None:
        if args.node_index == -1:
            focus_node = int(rng.integers(0, num_nodes))
            print(f"随机选择结点索引 {focus_node}。")
        else:
            focus_node = args.node_index
        if not (0 <= focus_node < num_nodes):
            raise ValueError(
                f"node-index 超出范围，需在 [0, {num_nodes - 1}] 之间或为 -1。"
            )
        i_idx, j_idx, weights, visible_nodes = focus_on_node(
            i_idx, j_idx, weights, focus_node
        )
        neighbor_count = np.unique(np.concatenate((i_idx, j_idx))).size
        print(
            f"中心结点 {focus_node} 连接 {neighbor_count} 个邻居。"
        )
    elif args.max_edges > 0 and weights.size > args.max_edges:
        i_idx, j_idx, weights = sample_edges(
            i_idx, j_idx, weights, args.max_edges, rng
        )
        print(f"随机下采样至 {weights.size:,} 条边用于绘图。")

    plot_graph(
        lat,
        lon,
        i_idx,
        j_idx,
        weights,
        args,
        visible_nodes=visible_nodes,
        focus_node=focus_node,
    )


if __name__ == "__main__":
    main()
