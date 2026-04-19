"""
main.py — 全流程统一入口

使用方法：
    cd finance/strategy
    python main.py                        # 默认参数运行（纯量化模式）
    python main.py --refresh              # 强制重新下载数据
    python main.py --top-n 4              # 调整持仓数量为4
    python main.py --no-chart             # 不生成图表
    python main.py --start 2018-01-01     # 自定义回测起始日
    python main.py --mode hybrid --advice # hybrid模式：LLM生成当日持仓建议
    python main.py --llm-backtest         # hybrid模式逐月回测（慎用，API消耗大）

完整流程：
    数据下载     →  ETL清洗
    ↓
    特征工程     →  动量因子 + ATR + 波动率 + 偏度 + OLS斜率/R² + 风格比值 + 跨境状态
    ↓
    信号生成     →  双引擎量化切换（进攻子策略+防守子策略）[quant] 或 DeepSeek双Agent [hybrid]
    ↓
    回测执行     →  T+1 + 成本扣除 + ATR仓位 + 熔断
    ↓
    绩效报告     →  年化/夏普/回撤/胜率/盈亏比
    ↓
    可视化       →  净值/回撤/热力图/轮动图
"""

import argparse
import logging
import sys
import os

# 确保 strategy 目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import (
    START_DATE, END_DATE, BENCHMARK_ETF,
    TOP_N, OUTPUT_DIR, SIGNAL_MODE, QUANT_ENGINE_MODE,
)
from data_fetcher      import load_data
from feature_engine    import build_features
from signal_generator  import (
    build_quant_anchor_weights,
    generate_monthly_holdings,
    generate_monthly_holdings_hybrid,
)
from backtest_engine   import run_backtest
from performance       import generate_report, latest_month_summary, format_latest_month_summary
from visualization     import generate_all_charts
from visualization     import make_output_dir_name
from advisor           import generate_advice, generate_pre_rebalance_advice, send_advice_email, send_email_message, load_email_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────
# 命令行参数解析
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="中国ETF双动量轮动策略回测")
    parser.add_argument("--refresh",    action="store_true", help="强制重新下载数据（忽略缓存）")
    parser.add_argument("--top-n",      type=int, default=TOP_N,       help="持仓ETF数量 (default: 5)")
    parser.add_argument("--start",      type=str, default=START_DATE,  help="回测起始日 (default: 2015-01-01)")
    parser.add_argument("--end",        type=str, default=END_DATE,    help="回测结束日 (default: 2026-03-31)")
    parser.add_argument("--capital",    type=float, default=1_000_000.0, help="初始资金（元）")
    parser.add_argument(
        "--mode", type=str, default=SIGNAL_MODE, choices=["quant", "hybrid"],
        help="信号模式: quant=纯量化(默认), hybrid=量化+DeepSeek LLM决策"
    )
    parser.add_argument(
        "--llm-backtest", action="store_true",
        help="hybrid模式下逐月调用LLM做历史回测（慎用：API调用量大）"
    )
    parser.add_argument("--no-chart",   action="store_true", help="不生成图表")
    parser.add_argument("--show-chart", action="store_true", help="运行时展示图表（需 GUI 环境）")
    # ── 投资建议与邮件 ──
    parser.add_argument("--advice",     action="store_true", help="回测后生成当日投资建议报告")
    parser.add_argument("--smtp-host",  type=str, default=None, help="SMTP服务器（优先级高于.env）")
    parser.add_argument("--smtp-port",  type=int, default=None, help="SMTP端口（优先级高于.env）")
    parser.add_argument("--smtp-user",  type=str, default=None, help="发件邮箱账号（优先级高于.env）")
    parser.add_argument("--smtp-pass",  type=str, default=None, help="邮箱授权码（优先级高于.env）")
    parser.add_argument("--email-from", type=str, default=None, help="发件人地址（优先级高于.env）")
    parser.add_argument("--email-to",   type=str, default=None, help="收件人地址（优先级高于.env）")
    parser.add_argument("--monthly-email", action="store_true", help="发送最近一个完整自然月收益邮件（cron 友好，仅在每月首个交易日发送）")
    parser.add_argument("--force-monthly-email", action="store_true", help="无视月份边界，强制发送最近一个完整自然月收益邮件")
    return parser.parse_args()


def is_first_trading_day_of_month(trading_dates: pd.DatetimeIndex) -> bool:
    """判断最新交易日是否是当月第一个交易日。"""
    if trading_dates is None or len(trading_dates) < 2:
        return False
    latest = trading_dates[-1]
    prev = trading_dates[-2]
    return latest.month != prev.month and latest.normalize() == pd.Timestamp.today().normalize()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    # 允许运行时覆盖部分 config 参数（不修改文件）
    import config as cfg
    cfg.TOP_N = args.top_n
    cfg.SIGNAL_MODE = args.mode  # 同步更新运行时模式

    use_hybrid = (args.mode == "hybrid") and (args.llm_backtest or args.advice)
    if args.mode == "hybrid" and args.llm_backtest:
        run_label = "hybrid_llm_backtest"
    elif args.mode == "hybrid":
        run_label = "hybrid_advice"
    else:
        run_label = "quant_backtest"
    run_output_dir = os.path.join(OUTPUT_DIR, make_output_dir_name(run_label))

    logger.info("=" * 55)
    logger.info("  中国ETF双动量轮动策略  |  全流程启动")
    logger.info("=" * 55)
    logger.info(f"  回测区间: {args.start} ~ {args.end}")
    logger.info(f"  持仓数量: Top-{args.top_n}")
    logger.info(f"  初始资金: {args.capital:,.0f} 元")
    logger.info(f"  信号模式: {args.mode.upper()}" + (" [LLM回测启用]" if args.llm_backtest else ""))

    # ── Step 1: 数据下载与ETL ──
    logger.info("\n[Step 1/5] 数据下载与ETL清洗...")
    close_matrix, ohlcv_panel = load_data(force_refresh=args.refresh)

    # 截断到回测区间
    close_matrix = close_matrix.loc[args.start:args.end]
    ohlcv_panel  = {
        sym: df.loc[args.start:args.end]
        for sym, df in ohlcv_panel.items()
    }

    # ── Step 2: 特征工程 ──
    logger.info("\n[Step 2/5] 特征工程（动量因子 + ATR + 波动率 + OLS + 风格比值）...")
    features = build_features(close_matrix, ohlcv_panel)

    # ── Step 3: 信号生成 ──
    llm_weights_by_date = None

    if use_hybrid and args.llm_backtest:
        logger.info("\n[Step 3/5] Hybrid模式：量化预选 + DeepSeek LLM逐月决策...")
        logger.warning("  ⚠️  LLM回测启用，将对每个调仓日调用DeepSeek API，请确认Token余额！")
        # 先生成量化持仓作为回退
        quant_holdings = generate_monthly_holdings(
            features["momentum_score"],
            features["close"],
            rsi=features.get("rsi"),
        )
        quant_weights_df = build_quant_anchor_weights(features, quant_holdings)
        # HMM市场状态（可选增强LLM上下文）
        market_states = None
        try:
            from market_state import compute_market_states
            logger.info("  计算HMM市场状态序列...")
            market_states = compute_market_states(close_matrix)
        except Exception as e:
            logger.warning(f"  HMM计算失败（将跳过）: {e}")

        monthly_holdings, llm_weights_by_date = generate_monthly_holdings_hybrid(
            close=close_matrix,
            features=features,
            quant_holdings=quant_holdings,
            quant_weights_df=quant_weights_df,
            market_states=market_states,
        )
        logger.info(f"  LLM调仓决策完成: {len(llm_weights_by_date)}个调仓日")
    else:
        if QUANT_ENGINE_MODE == "dual":
            logger.info("\n[Step 3/5] 双引擎量化信号生成（进攻子策略 + 防守子策略 + 月度切换）...")
        else:
            logger.info("\n[Step 3/5] 量化双动量信号生成（截面排名 + 绝对动量过滤）...")
        monthly_holdings = generate_monthly_holdings(
            features["momentum_score"],
            features["close"],
            rsi=features.get("rsi"),
        )

    # 打印最近3个月持仓预览
    etf_cols = [c for c in monthly_holdings.columns if not c.startswith("_")]
    recent = monthly_holdings[etf_cols].tail(3)
    held_preview = recent.apply(
        lambda row: ", ".join(row[row == 1].index.tolist()), axis=1
    )
    logger.info(f"  最近3个月目标持仓:\n{held_preview.to_string()}")

    # ── Step 4: 回测执行 ──
    logger.info("\n[Step 4/5] 回测执行（T+1 + ATR仓位 + 熔断风控）...")
    nav_df, trade_df = run_backtest(
        features=features,
        monthly_holdings=monthly_holdings,
        initial_capital=args.capital,
        llm_weights_by_date=llm_weights_by_date,  # None = 纯量化
    )

    # ── Step 5: 绩效报告 ──
    logger.info("\n[Step 5/5] 绩效报告生成...")

    # 基准：沪深300ETF 收盘价
    benchmark_close = None
    if BENCHMARK_ETF in close_matrix.columns:
        benchmark_close = close_matrix[BENCHMARK_ETF]

    metrics = generate_report(
        nav_df=nav_df,
        trade_df=trade_df,
        benchmark_close=benchmark_close,
        strategy_name=f"ETF双动量轮动策略-{args.mode.upper()} (Top-{args.top_n})",
    )

    # ── 可视化 ──
    if not args.no_chart:
        generate_all_charts(
            nav_df=nav_df,
            trade_df=trade_df,
            monthly_holdings=monthly_holdings,
            benchmark_close=benchmark_close,
            output_dir=run_output_dir,
            show=args.show_chart,
        )

    # ── 保存净值序列 ──
    os.makedirs(run_output_dir, exist_ok=True)
    nav_path   = os.path.join(run_output_dir, "nav_series.csv")
    trade_path = os.path.join(run_output_dir, "trade_log.csv")
    nav_df.to_csv(nav_path)
    if not trade_df.empty:
        trade_df.to_csv(trade_path, index=False)
    logger.info(f"净值序列 → {nav_path}")
    logger.info(f"交易记录 → {trade_path}")
    logger.info("全流程完成。")

    # ── Step 5b（可选）: 月度收益邮件 ──
    if args.monthly_email or args.force_monthly_email:
        logger.info("\n[Step 5b] 检查月度收益邮件...")

        env_email = load_email_config()
        smtp_host  = args.smtp_host  or env_email["smtp_host"]
        smtp_port  = args.smtp_port  or env_email["smtp_port"]
        smtp_user  = args.smtp_user  or env_email["smtp_user"]
        smtp_pass  = args.smtp_pass  or env_email["smtp_pass"]
        email_from = args.email_from or env_email["email_from"]
        email_to   = args.email_to   or env_email["email_to"]

        if not args.force_monthly_email and not is_first_trading_day_of_month(features["close"].index):
            logger.info("  当前不是本月首个交易日，跳过月度收益邮件")
        elif smtp_host and email_to:
            summary = latest_month_summary(nav_df, benchmark_close=benchmark_close, asof_date=features["close"].index[-1])
            report_text = format_latest_month_summary(summary, strategy_name=f"ETF双动量轮动策略 (Top-{args.top_n})")
            subject_month = summary["month_label"] if summary else pd.Timestamp.today().strftime("%Y-%m")
            send_email_message(
                report_text=report_text,
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                smtp_user=smtp_user,
                smtp_pass=smtp_pass,
                email_from=email_from or smtp_user,
                email_to=email_to,
                subject=f"ETF轮动策略 · 月度收益简报 · {subject_month}",
                html_title="ETF双动量轮动策略 · 月度收益简报",
            )
        else:
            logger.warning("  月度收益邮件未发送：SMTP_HOST 或 EMAIL_TO 未配置")

    # ── Step 6（可选）: 当日投资建议 ──
    if args.advice:
        logger.info("\n[Step 6] 生成当日投资建议...")

        # 读取邮箱配置：命令行参数 > .env 文件
        env_email = load_email_config()
        smtp_host  = args.smtp_host  or env_email["smtp_host"]
        smtp_port  = args.smtp_port  or env_email["smtp_port"]
        smtp_user  = args.smtp_user  or env_email["smtp_user"]
        smtp_pass  = args.smtp_pass  or env_email["smtp_pass"]
        email_from = args.email_from or env_email["email_from"]
        email_to   = args.email_to   or env_email["email_to"]

        advice = generate_advice(
            close_matrix=close_matrix,
            features=features,
            nav_df=nav_df,
            trade_df=trade_df,
            monthly_holdings=monthly_holdings,
            current_capital=args.capital,
            use_llm=(args.mode == "hybrid"),
        )

        # 判断今日是否为调仓信号日（调仓前一个交易日）
        from signal_generator import get_rebalance_dates as _get_rebal
        from advisor import is_pre_rebalance_day
        trading_dates = features["close"].index
        rebal_dates = _get_rebal(trading_dates)
        today_ts = trading_dates[-1]  # 最新交易日

        is_signal, exec_date = is_pre_rebalance_day(today_ts, trading_dates, rebal_dates)
        if is_signal and exec_date is not None:
            logger.info(f"  今日({today_ts.strftime('%Y-%m-%d')})是调仓信号日，"
                        f"明日({exec_date.strftime('%Y-%m-%d')})执行调仓")
            close_row = features["close"].loc[today_ts]
            # 获取当前持仓权重（从最近一次之前的调仓信号推算）
            prev_weights = {}
            if len(rebal_dates) >= 2:
                signal_cols = [c for c in monthly_holdings.columns if not c.startswith("_")]
                # 上一次调仓的持仓
                prev_idx = rebal_dates.get_loc(today_ts) if today_ts in rebal_dates else -1
                if prev_idx > 0:
                    prev_row = monthly_holdings[signal_cols].iloc[prev_idx - 1]
                    prev_held = prev_row[prev_row == 1].index.tolist()
                    if prev_held:
                        prev_weights = {s: 1.0 / len(prev_held) for s in prev_held}

            report_text = generate_pre_rebalance_advice(
                advice=advice,
                close_row=close_row,
                execution_date=exec_date,
                current_weights=prev_weights,
                current_capital=args.capital,
            )
        else:
            report_text = advice["report_text"]
            logger.info("  今日非调仓信号日，生成常规建议报告")

        print("\n" + report_text)

        # 发送邮件（smtp_host 和 email_to 均已配置时生效）
        if smtp_host and email_to:
            send_advice_email(
                report_text=report_text,
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                smtp_user=smtp_user,
                smtp_pass=smtp_pass,
                email_from=email_from or smtp_user,
                email_to=email_to,
            )
        elif email_to:
            logger.warning("提供了 email_to 但 SMTP_HOST 为空，跳过邮件发送")

    return metrics, nav_df, trade_df


if __name__ == "__main__":
    main()
