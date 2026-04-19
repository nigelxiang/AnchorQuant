"""
backtest_engine.py — 向量化回测引擎

关键设计：
- T+1 严格：月末最后一个交易日 收盘后产生信号，
  次月第一个交易日 开盘价 执行买卖
- 交易成本：买入+卖出各扣 (COMMISSION_RATE + SLIPPAGE_RATE)
- 仓位：由 risk_manager 的 ATR等风险预算 + 目标波动率缩放确定
- 熔断：CircuitBreaker 在每日 NAV 更新后实时检测
- 输出：逐日净值序列 + 调仓记录
"""

import numpy as np
import pandas as pd
import logging

from config import (
    COMMISSION_RATE, SLIPPAGE_RATE,
    TOP_N, DEFENSIVE_ETFS, TOTAL_INVEST_RATIO,
    BENCHMARK_ETF, TRADING_DAYS_PER_YEAR,
)
from signal_generator import get_rebalance_dates
from risk_manager import compute_position_weights, CircuitBreaker, apply_static_defensive_allocation
import config as _cfg

logger = logging.getLogger(__name__)


SINGLE_SIDE_COST = COMMISSION_RATE + SLIPPAGE_RATE   # 单边成本


# ─────────────────────────────────────────────
# 核心回测循环
# ─────────────────────────────────────────────
def run_backtest(
    features: dict,
    monthly_holdings: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    llm_weights_by_date: dict = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    向量化+逐月调仓回测。

    入参：
    - features:            build_features() 返回的字典
    - monthly_holdings:    generate_monthly_holdings() 返回的宽表 (date x symbol)
    - initial_capital:     初始资金（元）
    - llm_weights_by_date: {signal_date: {symbol: weight}}，hybrid模式由 LLM 提供权重；
                           为 None 时使用纯量化 ATR 权重（默认行为）。

    返回：
    - nav_series:        pd.DataFrame, columns=[nav, cash, invested, drawdown, is_circuit]
    - trade_log:         pd.DataFrame, 逐笔交易记录
    """
    close        = features["close"]
    atr_df       = features["atr"]
    hist_vol_df  = features["hist_vol"]
    momentum_df  = features["momentum_score"]

    trading_dates = close.index
    all_symbols   = close.columns.tolist()
    meta_cols     = [c for c in monthly_holdings.columns if c.startswith("_")]
    signal_cols   = [c for c in monthly_holdings.columns if not c.startswith("_")]

    # ── 初始化状态 ──
    cash        = float(initial_capital)  # 现金
    positions   = {}                       # {symbol: shares}
    nav         = float(initial_capital)
    scaler      = CircuitBreaker()

    nav_records   = []
    trade_records = []
    daily_rets    = []              # 追踪组合日收益率（用于实时vol计算）
    prev_nav      = float(initial_capital)
    last_target_weights = {}        # 最近一次月度调仓的目标权重
    day_count     = 0               # 自上次月度调仓起的交易日计数
    current_signal_engine = "single"
    dual_weekly_gate_active = False

    # 调仓日程表：信号日 → 目标持仓ETF列表（次月第一个交易日执行）
    rebal_signal_dates = monthly_holdings.index
    # 建立映射：信号日 → 执行日（信号日之后第一个交易日）
    execute_map = _build_execution_map(rebal_signal_dates, trading_dates)

    dual_gate_ma_series = None
    gate_ma_window = int(getattr(_cfg, "DUAL_WEEKLY_GATE_MA_WINDOW", 0))
    dual_gate_benchmark = getattr(_cfg, "DUAL_SWITCH_BENCHMARK", BENCHMARK_ETF)
    if gate_ma_window > 0 and dual_gate_benchmark in close.columns:
        dual_gate_ma_series = close[dual_gate_benchmark].shift(1).rolling(
            gate_ma_window,
            min_periods=max(10, gate_ma_window // 2),
        ).mean()

    # 按执行日建立待执行队列
    pending_rebal: dict[pd.Timestamp, pd.Timestamp] = {}   # {执行日: 信号日}
    for sig_date, exec_date in execute_map.items():
        pending_rebal[exec_date] = sig_date

    # ── 逐日回测 ──
    for today in trading_dates:
        # 1. 获取今日价格
        close_row    = close.loc[today]
        atr_row      = atr_df.loc[today] if today in atr_df.index else pd.Series(dtype=float)
        hist_vol_row = hist_vol_df.loc[today] if today in hist_vol_df.index else pd.Series(dtype=float)

        # 2. 计算当前 NAV（含 mark-to-market）
        portfolio_value = _calc_portfolio_value(positions, close_row)
        nav = cash + portfolio_value

        # 追踪日收益率
        if prev_nav > 0:
            daily_rets.append((nav - prev_nav) / prev_nav)
        prev_nav = nav
        day_count += 1

        # 3. 渐进式回撤检测
        is_scaled = scaler.update(nav, today)

        # 4. 若今日为执行日，执行调仓
        if today in pending_rebal:
            sig_date = pending_rebal[today]
            day_count = 0  # 重置日计数
            dual_weekly_gate_active = False
            current_signal_engine = str(monthly_holdings.loc[sig_date].get("_engine", "single"))

            # ── 权重来源：LLM hybrid 或 量化ATR  ──
            if llm_weights_by_date and sig_date in llm_weights_by_date:
                # Hybrid模式：直接使用LLM输出的权重（已归一化）
                target_weights = dict(llm_weights_by_date[sig_date])
                logger.debug(f"[Hybrid] 使用LLM权重: {sig_date.date()} → {list(target_weights.keys())}")
            else:
                # 纯量化模式：从月度持仓表读取目标ETF，ATR等风险预算计算权重
                target_holdings = _get_holdings_list(monthly_holdings, sig_date, signal_cols)
                target_weights = compute_position_weights(
                    holdings=target_holdings,
                    atr_row=atr_row,
                    close_row=close_row,
                    hist_vol_row=hist_vol_row,
                    circuit_breaker=None,
                )

            # 渐进式混合：按回撤深度混合风险仓位和防御仓位
            target_weights = scaler.blend_weights(target_weights)

            # 组合级实时vol缩放：用近N日组合实际vol替代个股vol估算
            vol_cap = getattr(_cfg, 'PORTFOLIO_VOL_CAP', 0)
            vol_lookback = getattr(_cfg, 'VOL_LOOKBACK', 20)
            if vol_cap > 0 and len(daily_rets) >= vol_lookback:
                realized_vol = np.std(daily_rets[-vol_lookback:]) * np.sqrt(TRADING_DAYS_PER_YEAR)
                if realized_vol > vol_cap:
                    vol_adj = vol_cap / realized_vol
                    target_weights = {s: w * vol_adj for s, w in target_weights.items()}

            target_weights = apply_static_defensive_allocation(target_weights)

            # 执行调仓（T+1 开盘价 ≈ 当日开盘；此处用 close 代替，可换 open_price）
            trades = _execute_rebalance(
                today=today,
                target_weights=target_weights,
                current_positions=positions,
                close_row=close_row,
                nav=nav,
                cash=cash,
            )
            for t in trades:
                trade_records.append(t)
                # 更新 cash 和 positions
                cash, positions = _apply_trade(t, cash, positions)

            # 调仓后重新计算 NAV（含成本）
            portfolio_value = _calc_portfolio_value(positions, close_row)
            nav = cash + portfolio_value
            last_target_weights = dict(target_weights)  # 记录本次调仓目标

        # 4b. 周中vol检查：每N个交易日检查组合实时vol，超标则减仓
        weekly_vol_cap  = getattr(_cfg, 'WEEKLY_VOL_CAP', 0)
        vol_check_freq  = getattr(_cfg, 'VOL_CHECK_FREQ', 5)
        # 周中vol使用更短的回溯窗口（更灵敏，提前发现急跌市）
        vol_lookback_w  = getattr(_cfg, 'WEEKLY_VOL_LOOKBACK', getattr(_cfg, 'VOL_LOOKBACK', 20))
        # 回撤速度检测参数
        dd_vel_days     = getattr(_cfg, 'DD_VELOCITY_DAYS', 10)
        dd_vel_trigger  = getattr(_cfg, 'DD_VELOCITY_TRIGGER', 0.0)   # 0=禁用

        # 判断是否需要做周中保护（vol过高 或 短期急跌）
        run_weekly_protect = (
            today not in pending_rebal          # 非调仓日
            and day_count > 0
            and day_count % vol_check_freq == 0
            and positions
        )

        if run_weekly_protect:
            protect_triggered = False
            vol_scale = 1.0

            # ── 1. 波动率检测 ──
            if weekly_vol_cap > 0 and len(daily_rets) >= vol_lookback_w:
                realized_vol = np.std(daily_rets[-vol_lookback_w:]) * np.sqrt(TRADING_DAYS_PER_YEAR)
                if realized_vol > weekly_vol_cap:
                    vol_scale = min(vol_scale, weekly_vol_cap / realized_vol)
                    protect_triggered = True

            # ── 2. 回撤速度检测（短期急跌保护）──
            if dd_vel_trigger > 0 and len(daily_rets) >= dd_vel_days:
                # 近 N 日累计收益率
                recent_rets = daily_rets[-dd_vel_days:]
                cumret = (1 + np.array(recent_rets)).prod() - 1
                if cumret < -dd_vel_trigger:
                    # 急跌：压缩仓位至 PORTFOLIO_VOL_CAP 比例
                    vol_cap_monthly = getattr(_cfg, 'PORTFOLIO_VOL_CAP', weekly_vol_cap)
                    # 用当前实际vol估算缩放比（如无足够数据，用 0.7 保守值）
                    if len(daily_rets) >= vol_lookback_w:
                        cur_vol = np.std(daily_rets[-vol_lookback_w:]) * np.sqrt(TRADING_DAYS_PER_YEAR)
                        dd_vol_scale = vol_cap_monthly / max(cur_vol, vol_cap_monthly)
                    else:
                        dd_vol_scale = 0.70
                    vol_scale = min(vol_scale, dd_vol_scale)
                    protect_triggered = True

            if protect_triggered and vol_scale < 0.999:
                reduce_weights = {}
                freed_value = 0.0
                for sym, shares in positions.items():
                    if sym not in DEFENSIVE_ETFS:
                        price = close_row.get(sym, np.nan)
                        if not pd.isna(price) and price > 0:
                            cur_val = shares * price
                            target_val = cur_val * vol_scale
                            if cur_val > target_val + nav * 0.02:
                                reduce_weights[sym] = target_val / nav
                                freed_value += (cur_val - target_val)
                            else:
                                reduce_weights[sym] = cur_val / nav
                        else:
                            reduce_weights[sym] = 0.0
                    else:
                        price = close_row.get(sym, np.nan)
                        if not pd.isna(price) and price > 0:
                            reduce_weights[sym] = shares * price / nav
                # 释放资金流向防御ETF
                vol_to_def = getattr(_cfg, 'VOL_SELL_TO_DEFENSE', False)
                if vol_to_def and freed_value > nav * 0.02:
                    def_budget = freed_value / nav
                    for ds in DEFENSIVE_ETFS:
                        reduce_weights[ds] = reduce_weights.get(ds, 0.0) + def_budget / len(DEFENSIVE_ETFS)
                if reduce_weights:
                    trades = _execute_rebalance(
                        today=today,
                        target_weights=reduce_weights,
                        current_positions=positions,
                        close_row=close_row,
                        nav=nav,
                        cash=cash,
                    )
                    for t in trades:
                        trade_records.append(t)
                        cash, positions = _apply_trade(t, cash, positions)
                    portfolio_value = _calc_portfolio_value(positions, close_row)
                    nav = cash + portfolio_value

        # 4c. 双引擎专属周级风险闸门：
        # 仅在当前月度信号来自进攻引擎时生效；
        # 一旦触发，只做一次减风险并保持到下次月调仓，避免周级频繁来回交易。
        dual_gate_enabled = bool(getattr(_cfg, "DUAL_WEEKLY_GATE_ENABLED", False))
        gate_dd_days = int(getattr(_cfg, "DUAL_WEEKLY_GATE_DD_DAYS", 0))
        gate_dd_trigger = float(getattr(_cfg, "DUAL_WEEKLY_GATE_DD_TRIGGER", 0.0))
        gate_scale = float(getattr(_cfg, "DUAL_WEEKLY_GATE_SCALE", 1.0))

        allow_dual_gate = (
            not llm_weights_by_date
            or bool(getattr(_cfg, "HYBRID_ENABLE_DUAL_WEEKLY_GATE", False))
        )

        run_dual_gate = (
            dual_gate_enabled
            and allow_dual_gate
            and current_signal_engine == "offensive"
            and not dual_weekly_gate_active
            and today not in pending_rebal
            and positions
        )
        if run_dual_gate:
            below_gate_ma = False
            if dual_gate_ma_series is not None and today in dual_gate_ma_series.index and pd.notna(dual_gate_ma_series.loc[today]):
                below_gate_ma = close_row.get(dual_gate_benchmark, np.nan) < dual_gate_ma_series.loc[today]

            fast_drop = False
            if gate_dd_days > 0 and gate_dd_trigger > 0 and len(daily_rets) >= gate_dd_days:
                recent_cumret = (1 + np.array(daily_rets[-gate_dd_days:])).prod() - 1
                fast_drop = recent_cumret < -gate_dd_trigger

            if (below_gate_ma or fast_drop) and gate_scale < 0.999:
                reduce_weights = {}
                freed_value = 0.0
                for sym, shares in positions.items():
                    price = close_row.get(sym, np.nan)
                    if pd.isna(price) or price <= 0:
                        continue
                    current_value = shares * price
                    if sym not in DEFENSIVE_ETFS:
                        target_value = current_value * gate_scale
                        reduce_weights[sym] = target_value / nav
                        freed_value += max(0.0, current_value - target_value)
                    else:
                        reduce_weights[sym] = current_value / nav

                if freed_value > nav * 0.005:
                    gate_pool = [
                        sym for sym in getattr(_cfg, "DUAL_ENGINE_DEFENSIVE_POOL", DEFENSIVE_ETFS)
                        if sym in close.columns
                    ] or DEFENSIVE_ETFS
                    def_budget = freed_value / nav
                    for ds in gate_pool:
                        reduce_weights[ds] = reduce_weights.get(ds, 0.0) + def_budget / len(gate_pool)

                if reduce_weights:
                    trades = _execute_rebalance(
                        today=today,
                        target_weights=reduce_weights,
                        current_positions=positions,
                        close_row=close_row,
                        nav=nav,
                        cash=cash,
                    )
                    for t in trades:
                        trade_records.append(t)
                        cash, positions = _apply_trade(t, cash, positions)
                    portfolio_value = _calc_portfolio_value(positions, close_row)
                    nav = cash + portfolio_value
                    dual_weekly_gate_active = True

        # 5. 计算当日回撤
        peak = scaler.peak_value or nav
        drawdown = (nav - peak) / peak if peak > 0 else 0.0

        nav_records.append({
            "date":         today,
            "nav":          nav,
            "cash":         cash,
            "invested":     portfolio_value,
            "drawdown":     drawdown,
            "is_circuit":   int(is_scaled),
        })

    nav_df   = pd.DataFrame(nav_records).set_index("date")
    trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

    return nav_df, trade_df


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────
def _build_execution_map(
    signal_dates: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
) -> dict:
    """
    对每个信号日，找到其下一个交易日作为执行日（T+1 语义）。
    """
    execute_map = {}
    trading_list = trading_dates.sort_values().tolist()
    td_set = set(trading_list)

    for sd in signal_dates:
        # 执行日 = 信号日后第一个交易日
        future = [d for d in trading_list if d > sd]
        if future:
            execute_map[sd] = future[0]
        else:
            # 信号在最后一个交易日，无法执行，跳过
            pass
    return execute_map


def _get_holdings_list(
    monthly_holdings: pd.DataFrame,
    date: pd.Timestamp,
    signal_cols: list,
) -> list:
    if date not in monthly_holdings.index:
        return DEFENSIVE_ETFS[:]
    row = monthly_holdings.loc[date, signal_cols]
    return row[row == 1].index.tolist()


def _calc_portfolio_value(positions: dict, close_row: pd.Series) -> float:
    total = 0.0
    for sym, shares in positions.items():
        price = close_row.get(sym, np.nan)
        if not pd.isna(price) and price > 0:
            total += shares * price
    return total


def _execute_rebalance(
    today: pd.Timestamp,
    target_weights: dict,
    current_positions: dict,
    close_row: pd.Series,
    nav: float,
    cash: float,
) -> list:
    """
    计算需要执行的买卖交易列表。
    单边成本 = SINGLE_SIDE_COST（买 or 卖各扣一次）。
    """
    trades = []
    target_values = {sym: nav * w for sym, w in target_weights.items()}

    # 当前持仓市值
    current_values = {}
    for sym, shares in current_positions.items():
        price = close_row.get(sym, np.nan)
        if not pd.isna(price) and price > 0:
            current_values[sym] = shares * price
        else:
            current_values[sym] = 0.0

    # 需要卖出的标的（目标权重=0或减少）
    for sym, cur_val in current_values.items():
        target_val = target_values.get(sym, 0.0)
        price = close_row.get(sym, np.nan)
        if pd.isna(price) or price <= 0:
            continue
        if cur_val > target_val + 1.0:   # 差额 > 1元才交易，避免微量调整
            sell_value = cur_val - target_val
            sell_shares = sell_value / price
            cost = sell_value * SINGLE_SIDE_COST
            trades.append({
                "date": today, "symbol": sym, "direction": "sell",
                "shares": sell_shares, "price": price,
                "value": sell_value, "cost": cost,
            })

    # 需要买入的标的（新增或增加）
    # 先结算卖出后的可用资金
    available_cash = cash + sum(
        t["value"] - t["cost"] for t in trades if t["direction"] == "sell"
    )
    for sym, target_val in target_values.items():
        cur_val = current_values.get(sym, 0.0)
        price = close_row.get(sym, np.nan)
        if pd.isna(price) or price <= 0:
            continue
        if target_val > cur_val + 1.0:
            buy_value = min(target_val - cur_val, available_cash * 0.999)
            if buy_value <= 0:
                continue
            buy_shares = buy_value / price
            cost = buy_value * SINGLE_SIDE_COST
            real_buy = buy_value - cost   # 实际买入价值（已扣成本）
            trades.append({
                "date": today, "symbol": sym, "direction": "buy",
                "shares": buy_shares, "price": price,
                "value": buy_value, "cost": cost,
            })
            available_cash -= buy_value

    return trades


def _apply_trade(trade: dict, cash: float, positions: dict) -> tuple[float, dict]:
    """将单笔交易应用到当前 cash 和 positions"""
    sym = trade["symbol"]
    shares = trade["shares"]
    value = trade["value"]
    cost = trade["cost"]

    if trade["direction"] == "sell":
        cash += value - cost
        positions[sym] = max(0.0, positions.get(sym, 0.0) - shares)
        if positions[sym] < 1e-6:
            del positions[sym]
    else:  # buy
        cash -= value
        positions[sym] = positions.get(sym, 0.0) + shares

    return cash, positions
