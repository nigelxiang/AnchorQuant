"""
visualization.py — 可视化模块
生成四张图表（保存为 PNG，同时可选 show）：
  1. 净值曲线 + 基准对比
  2. 回撤序列图（含熔断标记）
  3. 月度收益热力图
  4. 持仓轮动图（每月持仓各ETF的权重堆叠面积图）
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 无头环境（服务器）下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from config import OUTPUT_DIR, BENCHMARK_ETF, ETF_POOL

# 中文字体处理：优先使用系统中文字体，无则降级英文标签
import matplotlib.font_manager as fm
_cn_fonts = [f.name for f in fm.fontManager.ttflist if "CJK" in f.name or "Hei" in f.name
             or "Song" in f.name or "SimSun" in f.name or "WenQuanYi" in f.name]
if _cn_fonts:
    plt.rcParams["font.family"] = _cn_fonts[0]
else:
    # 无中文字体时，使用英文别名
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


# ─────────────────────────────────────────────
# 1. 净值曲线
# ─────────────────────────────────────────────
def plot_nav(
    nav_df: pd.DataFrame,
    benchmark_close: pd.Series = None,
    save_path: str = None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(14, 5))

    nav = nav_df["nav"] / nav_df["nav"].iloc[0]
    ax.plot(nav.index, nav.values, color="#1f77b4", linewidth=1.5, label="Strategy NAV")

    # 熔断区间标色
    if "is_circuit" in nav_df.columns:
        circuit_mask = nav_df["is_circuit"].astype(bool)
        ax.fill_between(
            nav_df.index, 0, nav.max() * 1.05,
            where=circuit_mask,
            color="red", alpha=0.08, label="Circuit Breaker Active",
        )

    # 基准净值
    if benchmark_close is not None:
        bench_aligned = benchmark_close.reindex(nav.index).ffill().dropna()
        bench_nav = bench_aligned / bench_aligned.iloc[0]
        ax.plot(bench_nav.index, bench_nav.values, color="#d62728",
                linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"Benchmark ({BENCHMARK_ETF} HS300 ETF)")

    ax.set_title("Strategy Net Asset Value vs Benchmark", fontsize=13)
    ax.set_ylabel("Normalized NAV (Start=1.0)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_or_show(fig, save_path or os.path.join(OUTPUT_DIR, "01_nav_curve.png"), show)


# ─────────────────────────────────────────────
# 2. 回撤序列图
# ─────────────────────────────────────────────
def plot_drawdown(
    nav_df: pd.DataFrame,
    save_path: str = None,
    show: bool = False,
):
    nav = nav_df["nav"]
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max * 100  # 转为百分比

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.6, label="Drawdown (%)")
    ax.axhline(-12, color="orange", linestyle="--", linewidth=1, label="Circuit Breaker Threshold (-12%)")
    ax.axhline(-15, color="red",    linestyle="--", linewidth=1, label="Hard Limit (-15%)")

    ax.set_title("Portfolio Drawdown (%)", fontsize=13)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_or_show(fig, save_path or os.path.join(OUTPUT_DIR, "02_drawdown.png"), show)


# ─────────────────────────────────────────────
# 3. 月度收益热力图
# ─────────────────────────────────────────────
def plot_monthly_heatmap(
    nav_df: pd.DataFrame,
    save_path: str = None,
    show: bool = False,
):
    nav = nav_df["nav"]
    monthly_ret = nav.resample("ME").last().pct_change().dropna() * 100  # 百分比

    # 转为 年×月 矩阵
    df = monthly_ret.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    cmap = LinearSegmentedColormap.from_list("rg", ["#d62728", "white", "#2ca02c"])

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.55)))
    im = ax.imshow(pivot.values, cmap=cmap, vmin=-10, vmax=10, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)

    # 在格子内写数值
    for i in range(len(pivot)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 6 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label="Monthly Return (%)")
    ax.set_title("Monthly Return Heatmap (%)", fontsize=13)
    plt.tight_layout()

    _save_or_show(fig, save_path or os.path.join(OUTPUT_DIR, "03_monthly_heatmap.png"), show)


# ─────────────────────────────────────────────
# 4. 持仓轮动堆叠图
# ─────────────────────────────────────────────
def plot_holding_rotation(
    monthly_holdings: pd.DataFrame,
    save_path: str = None,
    show: bool = False,
):
    """
    各月目标持仓（1/0）堆叠面积图（近似权重 = 等权展示）。
    """
    meta_cols = [c for c in monthly_holdings.columns if c.startswith("_")]
    etf_cols  = [c for c in monthly_holdings.columns if not c.startswith("_")]

    hold = monthly_holdings[etf_cols].copy()
    # 每行归一化（持有数量可能不等于 TOP_N）
    row_sum = hold.sum(axis=1).replace(0, 1)
    hold_pct = hold.div(row_sum, axis=0) * 100

    # 映射ETF代码到中文名（如有）
    rename_map = {k: f"{v}({k})" for k, v in ETF_POOL.items() if k in hold_pct.columns}
    hold_pct.rename(columns=rename_map, inplace=True)

    # 只保留出现过的列
    active_cols = hold_pct.columns[hold_pct.sum() > 0].tolist()
    hold_pct = hold_pct[active_cols]

    colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(14, 5))
    hold_pct.plot.area(ax=ax, stacked=True,
                       color=colors[:len(active_cols)],
                       linewidth=0, alpha=0.85)

    ax.set_title("Monthly Holding Rotation (Equal-Weight Approximation)", fontsize=13)
    ax.set_ylabel("Allocation (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    _save_or_show(fig, save_path or os.path.join(OUTPUT_DIR, "04_holding_rotation.png"), show)


# ─────────────────────────────────────────────
# 统一生成所有图表
# ─────────────────────────────────────────────
def generate_all_charts(
    nav_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    monthly_holdings: pd.DataFrame,
    benchmark_close: pd.Series = None,
    output_dir: str = None,
    show: bool = False,
):
    chart_dir = output_dir or OUTPUT_DIR
    print(f"\n生成图表，输出目录: {chart_dir}")
    plot_nav(
        nav_df,
        benchmark_close=benchmark_close,
        save_path=os.path.join(chart_dir, "01_nav_curve.png"),
        show=show,
    )
    print("  [1/4] 净值曲线 ✓")
    plot_drawdown(
        nav_df,
        save_path=os.path.join(chart_dir, "02_drawdown.png"),
        show=show,
    )
    print("  [2/4] 回撤序列 ✓")
    plot_monthly_heatmap(
        nav_df,
        save_path=os.path.join(chart_dir, "03_monthly_heatmap.png"),
        show=show,
    )
    print("  [3/4] 月度热力图 ✓")
    plot_holding_rotation(
        monthly_holdings,
        save_path=os.path.join(chart_dir, "04_holding_rotation.png"),
        show=show,
    )
    print("  [4/4] 持仓轮动图 ✓")
    print(f"图表已保存至: {chart_dir}\n")


def make_output_dir_name(label: str) -> str:
    """将运行标签转换为稳定的输出目录名。"""
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label).strip()).strip("_")
    return normalized or "run"


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def _save_or_show(fig, save_path: str, show: bool):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
