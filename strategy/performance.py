"""
performance.py — 绩效指标计算
输出完整的策略评估报告，包含：
  - 年化收益率
  - 最大回撤（含发生时间）
  - 夏普比率
  - 卡尔玛比率（年化收益 / 最大回撤）
  - 月度胜率
  - 盈亏比
  - 年换手率
"""

import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, BENCHMARK_ETF


# ─────────────────────────────────────────────
# 基础指标
# ─────────────────────────────────────────────
def annualized_return(nav_series: pd.Series) -> float:
    """年化收益率"""
    total_days = (nav_series.index[-1] - nav_series.index[0]).days
    if total_days <= 0:
        return 0.0
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    years = total_days / 365.25
    return (1 + total_return) ** (1 / years) - 1


def max_drawdown(nav_series: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    最大回撤及其发生区间。
    返回：(mdd值, 峰值日期, 谷值日期)，mdd为负数。
    """
    rolling_max = nav_series.cummax()
    drawdown = nav_series / rolling_max - 1
    mdd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = nav_series.loc[:trough_date].idxmax()
    return mdd, peak_date, trough_date


def sharpe_ratio(nav_series: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """年化夏普比率（使用日收益率）"""
    daily_returns = nav_series.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = daily_returns - daily_rf
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def calmar_ratio(ann_ret: float, mdd: float) -> float:
    """卡尔玛比率 = 年化收益 / |最大回撤|"""
    return ann_ret / abs(mdd) if mdd != 0 else np.inf


def sortino_ratio(nav_series: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """索提诺比率（只惩罚下行波动）"""
    daily_returns = nav_series.pct_change().dropna()
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess = daily_returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf
    return excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


# ─────────────────────────────────────────────
# 月度分析
# ─────────────────────────────────────────────
def monthly_win_rate(nav_series: pd.Series) -> tuple[float, float]:
    """
    月度胜率 & 盈亏比。
    返回：(胜率, 盈亏比)
    """
    monthly_ret = nav_series.resample("ME").last().pct_change().dropna()
    if monthly_ret.empty:
        return 0.0, 0.0

    wins  = monthly_ret[monthly_ret > 0]
    loses = monthly_ret[monthly_ret < 0]

    win_rate = len(wins) / len(monthly_ret)
    avg_win  = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = loses.mean() if len(loses) > 0 else -1e-9
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    return win_rate, profit_loss_ratio


# ─────────────────────────────────────────────
# 换手率
# ─────────────────────────────────────────────
def annual_turnover(trade_df: pd.DataFrame, nav_series: pd.Series) -> float:
    """
    年换手率 = 年均买入金额 / 平均净值
    """
    if trade_df.empty:
        return 0.0
    buy_trades = trade_df[trade_df["direction"] == "buy"]
    if buy_trades.empty:
        return 0.0

    buy_trades = buy_trades.copy()
    buy_trades["date"] = pd.to_datetime(buy_trades["date"])
    annual_buy = buy_trades.groupby(buy_trades["date"].dt.year)["value"].sum()
    avg_nav = nav_series.mean()

    if avg_nav == 0:
        return 0.0
    return (annual_buy / avg_nav).mean()


# ─────────────────────────────────────────────
# 基准比较（信息比率）
# ─────────────────────────────────────────────
def information_ratio(
    nav_series: pd.Series,
    benchmark_nav: pd.Series,
) -> float:
    """信息比率 = 超额收益均值 / 超额收益标准差 * sqrt(252)"""
    strat_ret = nav_series.pct_change().dropna()
    bench_ret = benchmark_nav.pct_change().dropna()
    # 对齐日期
    common = strat_ret.index.intersection(bench_ret.index)
    if len(common) < 10:
        return np.nan
    excess = strat_ret.loc[common] - bench_ret.loc[common]
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def latest_month_summary(
    nav_df: pd.DataFrame,
    benchmark_close: pd.Series = None,
    asof_date: pd.Timestamp = None,
) -> dict | None:
    """提取最近一个完整自然月的收益摘要。"""
    nav = nav_df["nav"].dropna()
    if len(nav) < 2:
        return None

    if asof_date is None:
        asof_date = nav.index[-1]
    asof_date = pd.Timestamp(asof_date)
    current_month_start = asof_date.replace(day=1)

    completed_nav = nav[nav.index < current_month_start]
    if completed_nav.empty:
        return None

    monthly_nav = completed_nav.resample("ME").last().dropna()
    if len(monthly_nav) < 2:
        return None

    month_end = monthly_nav.index[-1]
    prev_month_end = monthly_nav.index[-2]
    strategy_ret = monthly_nav.iloc[-1] / monthly_nav.iloc[-2] - 1

    benchmark_ret = np.nan
    excess_ret = np.nan
    if benchmark_close is not None:
        bench = benchmark_close.reindex(nav.index).ffill()
        completed_bench = bench[bench.index < current_month_start]
        monthly_bench = completed_bench.resample("ME").last().dropna()
        if len(monthly_bench) >= 2:
            benchmark_ret = monthly_bench.iloc[-1] / monthly_bench.iloc[-2] - 1
            excess_ret = strategy_ret - benchmark_ret

    return {
        "month_label": month_end.strftime("%Y-%m"),
        "month_start": (prev_month_end + pd.Timedelta(days=1)).date(),
        "month_end": month_end.date(),
        "month_start_nav": monthly_nav.iloc[-2],
        "month_end_nav": monthly_nav.iloc[-1],
        "strategy_return": strategy_ret,
        "benchmark_return": benchmark_ret,
        "excess_return": excess_ret,
    }


def format_latest_month_summary(summary: dict, strategy_name: str = "ETF双动量轮动策略") -> str:
    """将最近完整月收益摘要格式化为邮件正文。"""
    if not summary:
        return f"{strategy_name}：暂无足够数据生成最近一个月收益摘要。"

    lines = [
        "=" * 60,
        f"  {strategy_name} · 月度收益简报",
        "=" * 60,
        f"  统计月份    : {summary['month_label']}",
        f"  统计区间    : {summary['month_start']} ~ {summary['month_end']}",
        f"  月初净值    : {summary['month_start_nav']:,.0f}",
        f"  月末净值    : {summary['month_end_nav']:,.0f}",
        "-" * 60,
        f"  策略月收益  : {summary['strategy_return']:.2%}",
    ]

    if not np.isnan(summary["benchmark_return"]):
        lines.append(f"  基准月收益  : {summary['benchmark_return']:.2%}")
    if not np.isnan(summary["excess_return"]):
        lines.append(f"  超额收益    : {summary['excess_return']:.2%}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 主报告函数
# ─────────────────────────────────────────────
def generate_report(
    nav_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    benchmark_close: pd.Series = None,
    strategy_name: str = "ETF双动量轮动策略",
) -> dict:
    """
    生成完整绩效报告，返回指标字典，同时打印格式化报告。
    
    入参：
    - nav_df:          backtest_engine 返回的净值DataFrame（含 nav 列）
    - trade_df:        交易记录
    - benchmark_close: 基准ETF收盘价序列（用于计算超额收益）
    """
    nav = nav_df["nav"]

    ann_ret   = annualized_return(nav)
    mdd, p_dt, t_dt = max_drawdown(nav)
    sharpe    = sharpe_ratio(nav)
    calmar    = calmar_ratio(ann_ret, mdd)
    sortino   = sortino_ratio(nav)
    win_rate, pl_ratio = monthly_win_rate(nav)
    turnover  = annual_turnover(trade_df, nav)

    # 基准对比
    if benchmark_close is not None:
        bench_aligned = benchmark_close.reindex(nav.index).ffill()
        bench_nav = bench_aligned / bench_aligned.iloc[0] * nav.iloc[0]
        bench_ann_ret = annualized_return(bench_nav)
        bench_mdd, _, _ = max_drawdown(bench_nav)
        ir = information_ratio(nav, bench_nav)
        excess_return = ann_ret - bench_ann_ret
    else:
        bench_ann_ret = bench_mdd = ir = excess_return = float("nan")

    metrics = {
        "策略名称":     strategy_name,
        "回测区间":     f"{nav.index[0].date()} ~ {nav.index[-1].date()}",
        "初始资金":     f"{nav.iloc[0]:,.0f} 元",
        "最终净值":     f"{nav.iloc[-1]:,.0f} 元",
        "总收益率":     f"{nav.iloc[-1] / nav.iloc[0] - 1:.2%}",
        "年化收益率":   ann_ret,
        "基准年化收益": bench_ann_ret,
        "超额年化收益": excess_return,
        "最大回撤":     mdd,
        "回撤期间":     f"{p_dt.date()} → {t_dt.date()}",
        "基准最大回撤": bench_mdd,
        "夏普比率":     sharpe,
        "索提诺比率":   sortino,
        "卡尔玛比率":   calmar,
        "信息比率":     ir,
        "月度胜率":     win_rate,
        "盈亏比":       pl_ratio,
        "年均换手率":   turnover,
        "总交易笔数":   len(trade_df) if not trade_df.empty else 0,
    }

    # 格式化打印
    _print_report(metrics)

    return metrics


def _print_report(metrics: dict):
    print("\n" + "=" * 60)
    print(f"  {metrics['策略名称']} — 绩效报告")
    print("=" * 60)
    print(f"  回测区间    : {metrics['回测区间']}")
    print(f"  初始资金    : {metrics['初始资金']}")
    print(f"  最终净值    : {metrics['最终净值']}")
    print(f"  总收益率    : {metrics['总收益率']}")
    print("-" * 60)
    print(f"  年化收益率  : {metrics['年化收益率']:.2%}")
    print(f"  基准年化    : {metrics['基准年化收益']:.2%}" if not np.isnan(metrics['基准年化收益']) else "  基准年化    : N/A")
    print(f"  超额年化    : {metrics['超额年化收益']:.2%}" if not np.isnan(metrics['超额年化收益']) else "  超额年化    : N/A")
    print("-" * 60)
    print(f"  最大回撤    : {metrics['最大回撤']:.2%}  ({metrics['回撤期间']})")
    print(f"  基准最大回撤: {metrics['基准最大回撤']:.2%}" if not np.isnan(metrics['基准最大回撤']) else "  基准最大回撤: N/A")
    print("-" * 60)
    print(f"  夏普比率    : {metrics['夏普比率']:.3f}")
    print(f"  索提诺比率  : {metrics['索提诺比率']:.3f}")
    print(f"  卡尔玛比率  : {metrics['卡尔玛比率']:.3f}")
    print(f"  信息比率    : {metrics['信息比率']:.3f}" if not np.isnan(metrics['信息比率']) else "  信息比率    : N/A")
    print("-" * 60)
    print(f"  月度胜率    : {metrics['月度胜率']:.1%}")
    print(f"  盈亏比      : {metrics['盈亏比']:.2f}")
    print(f"  年均换手率  : {metrics['年均换手率']:.1%}")
    print(f"  总交易笔数  : {metrics['总交易笔数']}")
    print("=" * 60 + "\n")
