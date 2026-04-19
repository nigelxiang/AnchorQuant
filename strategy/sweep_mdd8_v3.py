"""
sweep_mdd8_v3.py — 渐进式风控（drawdown-based exposure scaling）
核心思路：
1. 根据当前回撤深度线性缩减风险仓位（不是二元熔断）
2. Emergency日内缩减（drawdown超过阈值时立即调仓）
3. 渐进恢复（drawdown缩小时逐步恢复仓位）
4. 测试不同频率的再平衡(月/双周/周)
"""
import sys, os, io, itertools, logging
import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
import risk_manager as rm
from data_fetcher import load_data
from feature_engine import build_features
from signal_generator import generate_monthly_holdings, get_rebalance_dates
from backtest_engine import (
    _build_execution_map, _get_holdings_list, _calc_portfolio_value,
    _execute_rebalance, _apply_trade
)
from risk_manager import calc_atr_weights, scale_to_target_vol
from performance import annualized_return, max_drawdown, sharpe_ratio, monthly_win_rate, sortino_ratio

print("Loading data...")
close_matrix, ohlcv_panel = load_data(force_refresh=False)
close_matrix = close_matrix.loc[cfg.START_DATE:cfg.END_DATE]
ohlcv_panel = {sym: df.loc[cfg.START_DATE:cfg.END_DATE] for sym, df in ohlcv_panel.items()}
print("Building features...")
features = build_features(close_matrix, ohlcv_panel)

# Generate holdings for different frequencies
print("Generating monthly holdings...")
old_stdout = sys.stdout
sys.stdout = io.StringIO()
monthly_holdings = generate_monthly_holdings(
    features["momentum_score"], features["close"], rsi=features.get("rsi")
)
sys.stdout = old_stdout

close = features["close"]
atr_df = features["atr"]
hist_vol_df = features["hist_vol"]
trading_dates = close.index
signal_cols = [c for c in monthly_holdings.columns if not c.startswith("_")]

# Build execution maps for different frequencies
def build_rebal_map(freq):
    """Build rebalance date → signal date mapping for given frequency."""
    if freq == "ME":
        rebal_dates = monthly_holdings.index
    else:
        # For biweekly/weekly, use closest prior monthly signal
        rebal_dates = get_rebalance_dates(trading_dates, freq=freq)
    
    exec_map = _build_execution_map(rebal_dates, trading_dates)
    pend = {}
    for sig, exe in exec_map.items():
        pend[exe] = sig
    return pend

pending_monthly = build_rebal_map("ME")
pending_biweekly = build_rebal_map("2W")
# For biweekly/weekly, map execution dates to nearest prior monthly signal
monthly_signal_dates = monthly_holdings.index.sort_values()

def get_latest_monthly_signal(date):
    """Find the most recent monthly signal date on or before `date`."""
    idx = monthly_signal_dates.searchsorted(date, side='right') - 1
    if idx >= 0:
        return monthly_signal_dates[idx]
    return None


def run_gradual_backtest(
    dd_start, dd_full, target_vol, tir,
    emergency_threshold, emergency_enabled=True,
    rebal_freq="ME",  # "ME", "2W"
    initial_capital=1_000_000.0
):
    """
    Backtest with gradual drawdown-based exposure scaling.
    
    dd_start: drawdown level where scaling begins (e.g., 0.02 = 2%)
    dd_full:  drawdown level where exposure = 0 (all defensive) (e.g., 0.07 = 7%)
    emergency_threshold: drawdown level that triggers immediate intraday rebalance
    target_vol: annual target volatility
    tir: total invest ratio (base level, before DD scaling)
    """
    cash = float(initial_capital)
    positions = {}
    nav = float(initial_capital)
    peak_value = float(initial_capital)
    
    nav_records = []
    trade_records = []
    last_emergency_date = None
    
    # Pick correct pending rebal map
    if rebal_freq == "2W":
        pending = pending_biweekly
    else:
        pending = pending_monthly
    
    for today in trading_dates:
        close_row = close.loc[today]
        atr_row = atr_df.loc[today] if today in atr_df.index else pd.Series(dtype=float)
        hist_vol_row = hist_vol_df.loc[today] if today in hist_vol_df.index else pd.Series(dtype=float)
        
        portfolio_value = _calc_portfolio_value(positions, close_row)
        nav = cash + portfolio_value
        
        # Update peak
        if nav > peak_value:
            peak_value = nav
        
        # Current drawdown
        dd = (peak_value - nav) / peak_value if peak_value > 0 else 0.0  # positive number
        
        # Compute exposure scaling factor based on drawdown depth
        if dd <= dd_start:
            dd_scale = 1.0  # Full exposure
        elif dd >= dd_full:
            dd_scale = 0.0  # All defensive
        else:
            dd_scale = 1.0 - (dd - dd_start) / (dd_full - dd_start)
        
        # ── Emergency rebalance ──
        if emergency_enabled and dd >= emergency_threshold and positions:
            # Only do emergency rebalance once per episode (not every day)
            if last_emergency_date is None or (today - last_emergency_date).days > 5:
                # Compute reduced-risk target based on dd_scale
                sig_date = get_latest_monthly_signal(today)
                if sig_date is not None:
                    target_holdings = _get_holdings_list(monthly_holdings, sig_date, signal_cols)
                else:
                    target_holdings = cfg.DEFENSIVE_ETFS[:]
                
                old_tir = rm.TOTAL_INVEST_RATIO
                rm.TOTAL_INVEST_RATIO = tir
                weights = calc_atr_weights(target_holdings, atr_row, close_row)
                weights = scale_to_target_vol(weights, hist_vol_row, target_vol=target_vol)
                rm.TOTAL_INVEST_RATIO = old_tir
                
                # Apply DD scaling
                if dd_scale < 0.01:
                    # Fully defensive
                    n_def = len(cfg.DEFENSIVE_ETFS)
                    weights = {sym: tir / n_def for sym in cfg.DEFENSIVE_ETFS}
                else:
                    # Blend: dd_scale * risk + (1-dd_scale) * defensive
                    n_def = len(cfg.DEFENSIVE_ETFS)
                    def_w = tir * (1 - dd_scale) / n_def
                    risk_weights = {sym: w * dd_scale for sym, w in weights.items()}
                    for ds in cfg.DEFENSIVE_ETFS:
                        risk_weights[ds] = risk_weights.get(ds, 0) + def_w
                    weights = risk_weights
                
                trades = _execute_rebalance(
                    today=today, target_weights=weights,
                    current_positions=positions, close_row=close_row,
                    nav=nav, cash=cash,
                )
                for t in trades:
                    trade_records.append(t)
                    cash, positions = _apply_trade(t, cash, positions)
                portfolio_value = _calc_portfolio_value(positions, close_row)
                nav = cash + portfolio_value
                last_emergency_date = today
        
        # ── Regular rebalance ──
        is_rebal_day = today in pending
        if is_rebal_day:
            if rebal_freq == "2W":
                sig_date = get_latest_monthly_signal(today)
            else:
                sig_date = pending[today]
            
            if sig_date is not None:
                target_holdings = _get_holdings_list(monthly_holdings, sig_date, signal_cols)
            else:
                target_holdings = cfg.DEFENSIVE_ETFS[:]
            
            old_tir = rm.TOTAL_INVEST_RATIO
            rm.TOTAL_INVEST_RATIO = tir
            weights = calc_atr_weights(target_holdings, atr_row, close_row)
            weights = scale_to_target_vol(weights, hist_vol_row, target_vol=target_vol)
            rm.TOTAL_INVEST_RATIO = old_tir
            
            # Apply DD scaling
            if dd_scale < 0.01:
                n_def = len(cfg.DEFENSIVE_ETFS)
                weights = {sym: tir / n_def for sym in cfg.DEFENSIVE_ETFS}
            else:
                n_def = len(cfg.DEFENSIVE_ETFS)
                def_w = tir * (1 - dd_scale) / n_def
                risk_weights = {sym: w * dd_scale for sym, w in weights.items()}
                for ds in cfg.DEFENSIVE_ETFS:
                    risk_weights[ds] = risk_weights.get(ds, 0) + def_w
                weights = risk_weights
            
            trades = _execute_rebalance(
                today=today, target_weights=weights,
                current_positions=positions, close_row=close_row,
                nav=nav, cash=cash,
            )
            for t in trades:
                trade_records.append(t)
                cash, positions = _apply_trade(t, cash, positions)
            portfolio_value = _calc_portfolio_value(positions, close_row)
            nav = cash + portfolio_value
        
        nav_records.append({"date": today, "nav": nav})
    
    nav_df = pd.DataFrame(nav_records).set_index("date")
    nav_s = nav_df["nav"]
    
    ann_ret = annualized_return(nav_s)
    mdd_val, _, _ = max_drawdown(nav_s)
    sharpe_val = sharpe_ratio(nav_s)
    sortino_val = sortino_ratio(nav_s)
    wr, _ = monthly_win_rate(nav_s)
    
    return ann_ret, mdd_val, sharpe_val, sortino_val, wr, len(trade_records)


# ─── Parameter space ───
# dd scaling configs
dd_configs = [
    # (dd_start, dd_full, emergency_threshold)
    (0.01, 0.05, 0.04),
    (0.01, 0.06, 0.04),
    (0.01, 0.06, 0.05),
    (0.01, 0.07, 0.05),
    (0.02, 0.05, 0.04),
    (0.02, 0.06, 0.04),
    (0.02, 0.06, 0.05),
    (0.02, 0.07, 0.05),
    (0.02, 0.07, 0.06),
    (0.02, 0.08, 0.06),
    (0.03, 0.06, 0.05),
    (0.03, 0.07, 0.05),
    (0.03, 0.07, 0.06),
    (0.03, 0.08, 0.06),
    (0.03, 0.08, 0.07),
]

tirs = [0.60, 0.70, 0.80, 0.90]
target_vols = [0.08, 0.10, 0.12, 0.15, 0.20]
freqs = ["ME", "2W"]

total = len(dd_configs) * len(tirs) * len(target_vols) * len(freqs)
print(f"Sweeping {total} configurations (gradual DD scaling)...\n")

results = []
count = 0

for (dds, ddf, emt), tir, tv, freq in itertools.product(dd_configs, tirs, target_vols, freqs):
    count += 1
    try:
        ann_ret, mdd, sharpe, sortino, wr, n_trades = run_gradual_backtest(
            dd_start=dds, dd_full=ddf, target_vol=tv, tir=tir,
            emergency_threshold=emt, emergency_enabled=True, rebal_freq=freq
        )
        results.append({
            "dds": dds, "ddf": ddf, "emt": emt, "tir": tir, "tv": tv, "freq": freq,
            "ann_ret": ann_ret, "mdd": mdd, "sharpe": sharpe,
            "sortino": sortino, "win_rate": wr, "trades": n_trades
        })
        if count % 200 == 0:
            print(f"  [{count}/{total}] ...")
    except Exception as e:
        print(f"  [{count}/{total}] ERROR: {e}")

df = pd.DataFrame(results)

# Filter MDD > -8%
valid = df[df["mdd"] > -0.08].copy()
print(f"\n{'='*110}")
print(f"Configs with MDD > -8%: {len(valid)} out of {len(df)}")
print(f"{'='*110}")

if len(valid) > 0:
    print(f"\nTOP-30 by Sharpe (MDD > -8%):")
    print("-"*110)
    top = valid.nlargest(30, "sharpe")
    for _, r in top.iterrows():
        print(f"  dds={r.dds:.2f} ddf={r.ddf:.2f} emt={r.emt:.2f} tir={r.tir:.2f} tv={r.tv:.2f} freq={r.freq} "
              f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
              f"sortino={r.sortino:.3f} wr={r.win_rate:.1%} tr={r.trades:.0f}")
    
    print(f"\nTOP-20 by Returns (MDD > -8%):")
    print("-"*110)
    top_ret = valid.nlargest(20, "ann_ret")
    for _, r in top_ret.iterrows():
        print(f"  dds={r.dds:.2f} ddf={r.ddf:.2f} emt={r.emt:.2f} tir={r.tir:.2f} tv={r.tv:.2f} freq={r.freq} "
              f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
              f"sortino={r.sortino:.3f} wr={r.win_rate:.1%}")

    print(f"\nBest Sharpe × Return (MDD > -8%):")
    print("-"*110)
    valid["score"] = valid["sharpe"] * valid["ann_ret"] * 100
    best_bal = valid.nlargest(20, "score")
    for _, r in best_bal.iterrows():
        print(f"  dds={r.dds:.2f} ddf={r.ddf:.2f} emt={r.emt:.2f} tir={r.tir:.2f} tv={r.tv:.2f} freq={r.freq} "
              f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
              f"sortino={r.sortino:.3f} wr={r.win_rate:.1%}")

# Compare to best binary CB result
print(f"\n{'='*110}")
print("Comparison: Best binary CB → cb=0.050, tir=0.80, tv=0.12, cbt=3")
print("  → ret=6.15%, mdd=-7.24%, sharpe=0.647, sortino=0.770")
print(f"{'='*110}")

# Pareto frontier
print(f"\nPareto Frontier (best Sharpe at each MDD bucket):")
print("-"*110)
df["mdd_bucket"] = (df["mdd"] * 200).round(0) / 2
pareto = df.loc[df.groupby("mdd_bucket")["sharpe"].idxmax()]
pareto = pareto.sort_values("mdd", ascending=False).head(20)
for _, r in pareto.iterrows():
    tag = "★" if r.mdd > -0.08 else " "
    print(f"  {tag} mdd={r.mdd:.2%} → dds={r.dds:.2f} ddf={r.ddf:.2f} emt={r.emt:.2f} tir={r.tir:.2f} "
          f"tv={r.tv:.2f} freq={r.freq} ret={r.ann_ret:.2%} sharpe={r.sharpe:.3f}")
