#!/usr/bin/env python3
"""统计 processed/*/meta.csv 中各列取值分布并输出图表与数据."""

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from pandas.api.types import CategoricalDtype

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="统计 processed/*/meta.csv 的列分布并生成图表与统计文件"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "data" / "processed",
        help="processed 目录路径，默认指向仓库 data/processed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "results" / "meta_eda",
        help="输出根目录，将按照数据集拆分子文件夹",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="连续数值绘制直方图时的最大分箱数",
    )
    parser.add_argument(
        "--unique-threshold",
        type=int,
        default=20,
        help="判定列为离散类型的最大唯一值数阈值",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.05,
        help="唯一值占比低于该阈值时按离散列处理",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=40,
        help="绘制离散列图表时展示的最多类别数量",
    )
    return parser.parse_args()


def find_meta_files(data_root: Path) -> List[Path]:
    """检索 data_root 下的所有 meta.csv."""
    return sorted(data_root.glob("*/meta.csv"))


def is_discrete(series: pd.Series, unique_threshold: int, ratio_threshold: float) -> bool:
    """根据 dtype 与唯一值密度判定是否为离散列."""
    if ptypes.is_bool_dtype(series) or isinstance(series.dtype, CategoricalDtype):
        return True
    if ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series):
        return True
    unique_count = series.nunique(dropna=True)
    total = series.notna().sum()
    if total == 0:
        return True
    if unique_count <= unique_threshold:
        return True
    return unique_count / total <= ratio_threshold


def plot_discrete(
    counts: pd.Series, dataset: str, column: str, plot_dir: Path, limit: int
) -> Path:
    """绘制离散列频次柱状图."""
    display_counts = counts.head(limit)
    display_counts.index = display_counts.index.astype(str)
    truncated = len(counts) > limit
    fig, ax = plt.subplots(figsize=(max(6, len(display_counts) * 0.6), 4))
    display_counts.plot(kind="bar", ax=ax, color="#1f77b4")
    title = f"{dataset}-{column} 离散分布"
    if truncated:
        title += f"(前{limit}项)"
    ax.set_title(title)
    ax.set_xlabel("取值")
    ax.set_ylabel("频次")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    plot_path = plot_dir / f"{column}_discrete.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def plot_continuous(series: pd.Series, dataset: str, column: str, plot_dir: Path, bins: int) -> Path:
    """绘制连续列直方图."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series, bins=bins, color="#ff7f0e", edgecolor="black")
    ax.set_title(f"{dataset}-{column} 直方图")
    ax.set_xlabel(column)
    ax.set_ylabel("频次")
    fig.tight_layout()
    plot_path = plot_dir / f"{column}_hist.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def analyze_dataset(
    meta_path: Path,
    output_root: Path,
    bins: int,
    unique_threshold: int,
    ratio_threshold: float,
    max_categories: int,
) -> dict:
    """处理单个 meta.csv 并返回统计摘要."""
    dataset_name = meta_path.parent.name
    df = pd.read_csv(meta_path)
    dataset_dir = output_root / dataset_name
    table_dir = dataset_dir / "tables"
    plot_dir = dataset_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    column_summaries = []

    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        non_null = series.dropna()
        if non_null.empty:
            column_summaries.append(
                {
                    "column": column,
                    "type": "empty",
                    "non_null": 0,
                    "missing": missing,
                }
            )
            continue

        discrete = is_discrete(non_null, unique_threshold, ratio_threshold)

        if discrete:
            freq_series = non_null.astype(str).value_counts()
            freq_path = table_dir / f"{column}_freq.csv"
            freq_series.to_csv(freq_path, header=["count"], encoding="utf-8")
            plot_path = plot_discrete(freq_series, dataset_name, column, plot_dir, max_categories)
            column_summaries.append(
                {
                    "column": column,
                    "type": "discrete",
                    "non_null": int(freq_series.sum()),
                    "missing": missing,
                    "unique": int(freq_series.shape[0]),
                    "table": str(freq_path),
                    "plot": str(plot_path),
                    "top_values": freq_series.head(min(5, len(freq_series))).to_dict(),
                }
            )
            continue

        numeric_series = pd.to_numeric(non_null, errors="coerce").dropna()
        if numeric_series.empty:
            # 假如无法转换为数值则退化为离散处理
            freq_series = non_null.astype(str).value_counts()
            freq_path = table_dir / f"{column}_freq.csv"
            freq_series.to_csv(freq_path, header=["count"], encoding="utf-8")
            plot_path = plot_discrete(freq_series, dataset_name, column, plot_dir, max_categories)
            column_summaries.append(
                {
                    "column": column,
                    "type": "discrete_fallback",
                    "non_null": int(freq_series.sum()),
                    "missing": missing,
                    "unique": int(freq_series.shape[0]),
                    "table": str(freq_path),
                    "plot": str(plot_path),
                    "top_values": freq_series.head(min(5, len(freq_series))).to_dict(),
                }
            )
            continue

        actual_bins = max(1, min(bins, int(numeric_series.nunique())))
        hist_counts, bin_edges = np.histogram(numeric_series, bins=actual_bins)
        hist_df = pd.DataFrame(
            {
                "bin_left": bin_edges[:-1],
                "bin_right": bin_edges[1:],
                "count": hist_counts,
            }
        )
        hist_path = table_dir / f"{column}_hist.csv"
        hist_df.to_csv(hist_path, index=False, encoding="utf-8")
        plot_path = plot_continuous(numeric_series, dataset_name, column, plot_dir, actual_bins)
        desc = numeric_series.describe().to_dict()
        desc["median"] = float(numeric_series.median())
        desc = {k: float(v) for k, v in desc.items()}
        column_summaries.append(
            {
                "column": column,
                "type": "continuous",
                "non_null": int(numeric_series.shape[0]),
                "missing": missing,
                "unique": int(numeric_series.nunique()),
                "table": str(hist_path),
                "plot": str(plot_path),
                "statistics": desc,
            }
        )

    summary = {
        "dataset": dataset_name,
        "source": str(meta_path),
        "columns": column_summaries,
    }
    summary_path = dataset_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """脚本入口."""
    args = parse_args()
    meta_files = find_meta_files(args.data_root)
    if not meta_files:
        print(f"在 {args.data_root} 下没有找到 meta.csv")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    index = []
    for meta_file in meta_files:
        print(f"处理 {meta_file} ...")
        summary = analyze_dataset(
            meta_file,
            args.output_root,
            args.bins,
            args.unique_threshold,
            args.ratio_threshold,
            args.max_categories,
        )
        index.append(summary)

    index_path = args.output_root / "meta_overview.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"所有处理完成，汇总写入 {index_path}")


if __name__ == "__main__":
    main()
