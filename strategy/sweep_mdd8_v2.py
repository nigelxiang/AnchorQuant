"""
sweep_mdd8_v2.py — 修正TARGET_VOL覆盖 + 更精细网格 + CB超时调参
"""
import sys, os, io, itertools, logging
import numpy as np
import pandas as pd

# Suppress all logging
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
import risk_manager as rm
from data_fetcher import load_data
from feature_engine import build_features
from signal_generator import generate_monthly_holdings
from backtest_engine import (
    _build_execution_map, _get_holdings_list, _calc_portfolio_value,
    _execute_rebalance, _apply_trade, SINGLE_SIDE_COST
)
from risk_manager import compute_position_weights, calc_atr_weights, scale_to_target_vol, CircuitBreaker
from performance import annualized_return, max_drawdown, sharpe_ratio, monthly_win_rate, sortino_ratio

# ─── Load data & features once ───
print("Loading data...")
close_matrix, ohlcv_panel = load_data(force_refresh=False)
close_matrix = close_matrix.loc[cfg.START_DATE:cfg.END_DATE]
ohlcv_panel = {sym: df.loc[cfg.START_DATE:cfg.END_DATE] for sym, df in ohlcv_panel.items()}
print("Building features...")
features = build_features(close_matrix, ohlcv_panel)
print("Generating monthly holdings...")
old_stdout = sys.stdout
sys.stdout = io.StringIO()
monthly_holdings = generate_monthly_holdings(
    features["momentum_score"], features["close"], rsi=features.get("rsi")
)
sys.stdout = old_stdout
print(f"Data ready. {len(close_matrix)} trading days.\n")

close = features["close"]
atr_df = features["atr"]
hist_vol_df = features["hist_vol"]
trading_dates = close.index
signal_cols = [c for c in monthly_holdings.columns if not c.startswith("_")]
rebal_signal_dates = monthly_holdings.index
execute_map = _build_execution_map(rebal_signal_dates, trading_dates)
pending_rebal = {}
for sig_date, exec_date in execute_map.items():
    pending_rebal[exec_date] = sig_date


def run_modified_backtest(
    cb_trigger, tir, target_vol, emergency_rebal=True, cb_timeout=3,
    initial_capital=1_000_000.0
):
    """Modified backtest with emergency rebalance + configurable CB timeout."""
    cash = float(initial_capital)
    positions = {}
    nav = float(initial_capital)
    scaler = CircuitBreaker(trigger=cb_trigger)
    
    nav_records = []
    trade_records = []
    
    for today in trading_dates:
        close_row = close.loc[today]
        atr_row = atr_df.loc[today] if today in atr_df.index else pd.Series(dtype=float)
        hist_vol_row = hist_vol_df.loc[today] if today in hist_vol_df.index else pd.Series(dtype=float)
        
        portfolio_value = _calc_portfolio_value(positions, close_row)
        nav = cash + portfolio_value
        
        was_triggered = scaler.is_triggered
        
        # Custom CB timeout
        if scaler.is_triggered and scaler.trigger_date is not None:
            months = (
                (today.year - scaler.trigger_date.year) * 12
                + (today.month - scaler.trigger_date.month)
            )
            if months >= cb_timeout:
                scaler.is_triggered = False
                scaler.trigger_date = None
                scaler.peak_value = nav
        
        # Update peak
        if not scaler.is_triggered and nav > (scaler.peak_value or 0):
            scaler.peak_value = nav
        
        # DD detection  
        if scaler.peak_value and scaler.peak_value > 0:
            dd = (nav - scaler.peak_value) / scaler.peak_value
        else:
            dd = 0.0
            
        if not scaler.is_triggered and dd <= -cb_trigger:
            scaler.is_triggered = True
            scaler.trigger_date = today
            
        # Recovery
        if scaler.is_triggered and scaler.peak_value and nav >= scaler.peak_value * (1 - cb_trigger * 0.5):
            scaler.is_triggered = False
            scaler.trigger_date = None
            scaler.peak_value = nav
        
        is_scaled = scaler.is_triggered
        just_triggered = is_scaled and not was_triggered
        
        # ── Emergency rebalance on CB trigger day ──
        if emergency_rebal and just_triggered and positions:
            defensive_etfs = cfg.DEFENSIVE_ETFS
            n_def = len(defensive_etfs)
            emergency_weights = {sym: tir / n_def for sym in defensive_etfs}
            
            trades = _execute_rebalance(
                today=today, target_weights=emergency_weights,
                current_positions=positions, close_row=close_row,
                nav=nav, cash=cash,
            )
            for t in trades:
                trade_records.append(t)
                cash, positions = _apply_trade(t, cash, positions)
            portfolio_value = _calc_portfolio_value(positions, close_row)
            nav = cash + portfolio_value
        
        # ── Regular monthly rebalance ──
        if today in pending_rebal:
            sig_date = pending_rebal[today]
            target_holdings = _get_holdings_list(monthly_holdings, sig_date, signal_cols)
            
            # ATR risk parity with correct param overrides
            old_tir = rm.TOTAL_INVEST_RATIO
            old_tv = rm.TARGET_VOL
            rm.TOTAL_INVEST_RATIO = tir
            rm.TARGET_VOL = target_vol
            
            weights = calc_atr_weights(target_holdings, atr_row, close_row)
            weights = scale_to_target_vol(weights, hist_vol_row, target_vol=target_vol)
            
            rm.TOTAL_INVEST_RATIO = old_tir
            rm.TARGET_VOL = old_tv
            
            # CB blend
            if is_scaled:
                n_def = len(cfg.DEFENSIVE_ETFS)
                weights = {sym: tir / n_def for sym in cfg.DEFENSIVE_ETFS}
            
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
        
        peak = scaler.peak_value or nav
        drawdown = (nav - peak) / peak if peak > 0 else 0.0
        nav_records.append({"date": today, "nav": nav})
    
    nav_df = pd.DataFrame(nav_records).set_index("date")
    nav_s = nav_df["nav"]
    
    ann_ret = annualized_return(nav_s)
    mdd_val, _, _ = max_drawdown(nav_s)
    sharpe_val = sharpe_ratio(nav_s)
    sortino_val = sortino_ratio(nav_s)
    wr, _ = monthly_win_rate(nav_s)
    
    return ann_ret, mdd_val, sharpe_val, sortino_val, wr, len(trade_records)


# ─── Fine-grained parameter space ───
cb_triggers = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07]
tirs = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
target_vols = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
cb_timeouts = [1, 2, 3]

# Only sweep with emergency_rebal=True (key for low MDD) 
total = len(cb_triggers) * len(tirs) * len(target_vols) * len(cb_timeouts)
print(f"Sweeping {total} configurations (emergency rebalance ON)...\n")

results = []
count = 0

for cb, tir, tv, cbt in itertools.product(cb_triggers, tirs, target_vols, cb_timeouts):
    count += 1
    try:
        ann_ret, mdd, sharpe, sortino, wr, n_trades = run_modified_backtest(
            cb_trigger=cb, tir=tir, target_vol=tv, emergency_rebal=True, cb_timeout=cbt
        )
        results.append({
            "cb": cb, "tir": tir, "tv": tv, "cbt": cbt,
            "ann_ret": ann_ret, "mdd": mdd, "sharpe": sharpe,
            "sortino": sortino, "win_rate": wr, "trades": n_trades
        })
        if count % 200 == 0:
            print(f"  [{count}/{total}] ...")
    except Exception as e:
        print(f"  [{count}/{total}] ERROR: {e}")

df = pd.DataFrame(results)

# ─── Filter MDD > -8% ───
valid = df[df["mdd"] > -0.08].copy()
print(f"\n{'='*100}")
print(f"Configs with MDD > -8%: {len(valid)} out of {len(df)}")
print(f"{'='*100}")

if len(valid) > 0:
    print(f"\nTOP-30 by Sharpe (MDD > -8%):")
    print("-"*100)
    top = valid.nlargest(30, "sharpe")
    for _, r in top.iterrows():
        print(f"  cb={r.cb:.3f} tir={r.tir:.2f} tv={r.tv:.2f} cbt={r.cbt:.0f} "
              f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
              f"sortino={r.sortino:.3f} wr={r.win_rate:.1%} trades={r.trades:.0f}")
    
    print(f"\nTOP-30 by Returns (MDD > -8%):")
    print("-"*100)
    top_ret = valid.nlargest(30, "ann_ret")
    for _, r in top_ret.iterrows():
        print(f"  cb={r.cb:.3f} tir={r.tir:.2f} tv={r.tv:.2f} cbt={r.cbt:.0f} "
              f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
              f"sortino={r.sortino:.3f} wr={r.win_rate:.1%}")

    print(f"\nBest balance (Sharpe×Return, MDD > -8%):")
    print("-"*100)
    valid["score"] = valid["sharpe"] * valid["ann_ret"] * 100
    best_bal = valid.nlargest(20, "score")
    for _, r in best_bal.iterrows():
        print(f"  cb={r.cb:.3f} tir={r.tir:.2f} tv={r.tv:.2f} cbt={r.cbt:.0f} "
              f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
              f"sortino={r.sortino:.3f} wr={r.win_rate:.1%}")

# Pareto frontier
print(f"\n{'='*100}")
print("Full Pareto Frontier (best Sharpe at each MDD bucket):")
print(f"{'='*100}")
df["mdd_bucket"] = (df["mdd"] * 200).round(0) / 2  # 0.5% buckets
pareto = df.loc[df.groupby("mdd_bucket")["sharpe"].idxmax()]
pareto = pareto.sort_values("mdd", ascending=False).head(40)
for _, r in pareto.iterrows():
    tag = "★" if r.mdd > -0.08 else " "
    print(f"  {tag} mdd={r.mdd:.2%} → cb={r.cb:.3f} tir={r.tir:.2f} tv={r.tv:.2f} cbt={r.cbt:.0f} "
          f"ret={r.ann_ret:.2%} sharpe={r.sharpe:.3f} sortino={r.sortino:.3f}")
