"""
sweep_mdd8.py — 大规模参数扫描：目标MDD<8%
核心新增：紧急日内再平衡（CB触发当天立即切换到防御仓位）
"""
import sys, os, io, itertools
import numpy as np
import pandas as pd

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
from risk_manager import compute_position_weights, CircuitBreaker
from performance import annualized_return, max_drawdown, sharpe_ratio, monthly_win_rate, sortino_ratio

# ─── Load data & features once ───
print("Loading data...")
close_matrix, ohlcv_panel = load_data(force_refresh=False)
close_matrix = close_matrix.loc[cfg.START_DATE:cfg.END_DATE]
ohlcv_panel = {sym: df.loc[cfg.START_DATE:cfg.END_DATE] for sym, df in ohlcv_panel.items()}
print("Building features...")
features = build_features(close_matrix, ohlcv_panel)
print("Generating monthly holdings...")
# Generate monthly_holdings once (RSI/TOP_N unchanged)
old_stdout = sys.stdout
sys.stdout = io.StringIO()
monthly_holdings = generate_monthly_holdings(
    features["momentum_score"], features["close"], rsi=features.get("rsi")
)
sys.stdout = old_stdout
print(f"Data ready. {len(close_matrix)} trading days, {len(monthly_holdings)} rebalance dates.\n")

# Pre-compute constants
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
    cb_trigger, tir, target_vol, emergency_rebal=True, 
    initial_capital=1_000_000.0
):
    """
    Modified backtest with emergency intra-day rebalance on CB trigger.
    Returns (ann_ret, mdd, sharpe, sortino, win_rate)
    """
    cash = float(initial_capital)
    positions = {}
    nav = float(initial_capital)
    
    # Create circuit breaker with custom trigger
    scaler = CircuitBreaker(trigger=cb_trigger)
    
    nav_records = []
    trade_records = []
    prev_cb_state = False  # Track CB state for emergency rebalance detection
    
    for today in trading_dates:
        close_row = close.loc[today]
        atr_row = atr_df.loc[today] if today in atr_df.index else pd.Series(dtype=float)
        hist_vol_row = hist_vol_df.loc[today] if today in hist_vol_df.index else pd.Series(dtype=float)
        
        # Calculate NAV
        portfolio_value = _calc_portfolio_value(positions, close_row)
        nav = cash + portfolio_value
        
        # Update circuit breaker
        was_triggered = scaler.is_triggered
        is_scaled = scaler.update(nav, today)
        just_triggered = is_scaled and not was_triggered
        
        # ── Emergency rebalance on CB trigger day ──
        if emergency_rebal and just_triggered and positions:
            # Immediately switch to defensive positions
            defensive_etfs = cfg.DEFENSIVE_ETFS
            n_def = len(defensive_etfs)
            emergency_weights = {sym: tir / n_def for sym in defensive_etfs}
            
            trades = _execute_rebalance(
                today=today,
                target_weights=emergency_weights,
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
        
        # ── Regular monthly rebalance ──
        if today in pending_rebal:
            sig_date = pending_rebal[today]
            target_holdings = _get_holdings_list(monthly_holdings, sig_date, signal_cols)
            
            # ATR risk parity with custom TIR and TARGET_VOL
            # Temporarily override config values
            old_tir = cfg.TOTAL_INVEST_RATIO
            old_tv = cfg.TARGET_VOL
            rm_old_tir = rm.TOTAL_INVEST_RATIO
            cfg.TOTAL_INVEST_RATIO = tir
            cfg.TARGET_VOL = target_vol
            rm.TOTAL_INVEST_RATIO = tir
            
            target_weights = compute_position_weights(
                holdings=target_holdings,
                atr_row=atr_row,
                close_row=close_row,
                hist_vol_row=hist_vol_row,
                circuit_breaker=None,
            )
            
            # Restore
            cfg.TOTAL_INVEST_RATIO = old_tir
            cfg.TARGET_VOL = old_tv
            rm.TOTAL_INVEST_RATIO = rm_old_tir
            
            # Apply CB blend
            if is_scaled:
                n_def = len(cfg.DEFENSIVE_ETFS)
                target_weights = {sym: tir / n_def for sym in cfg.DEFENSIVE_ETFS}
            
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
                cash, positions = _apply_trade(t, cash, positions)
            
            portfolio_value = _calc_portfolio_value(positions, close_row)
            nav = cash + portfolio_value
        
        peak = scaler.peak_value or nav
        drawdown = (nav - peak) / peak if peak > 0 else 0.0
        
        nav_records.append({
            "date": today,
            "nav": nav,
            "is_circuit": int(is_scaled),
        })
    
    nav_df = pd.DataFrame(nav_records).set_index("date")
    nav_s = nav_df["nav"]
    
    ann_ret = annualized_return(nav_s)
    mdd_val, _, _ = max_drawdown(nav_s)
    sharpe_val = sharpe_ratio(nav_s)
    sortino_val = sortino_ratio(nav_s)
    wr, _ = monthly_win_rate(nav_s)
    n_trades = len(trade_records)
    
    return ann_ret, mdd_val, sharpe_val, sortino_val, wr, n_trades


# ─── Parameter space ───
cb_triggers = [0.04, 0.05, 0.06, 0.07, 0.08]
tirs = [0.50, 0.60, 0.70, 0.80, 0.90, 0.93]
target_vols = [0.08, 0.10, 0.12, 0.15, 0.20]
emergency_options = [True, False]

total = len(cb_triggers) * len(tirs) * len(target_vols) * len(emergency_options)
print(f"Sweeping {total} configurations...\n")

results = []
count = 0

for cb, tir, tv, emerg in itertools.product(cb_triggers, tirs, target_vols, emergency_options):
    count += 1
    try:
        ann_ret, mdd, sharpe, sortino, wr, n_trades = run_modified_backtest(
            cb_trigger=cb, tir=tir, target_vol=tv, emergency_rebal=emerg
        )
        results.append({
            "cb": cb, "tir": tir, "tv": tv, "emerg": emerg,
            "ann_ret": ann_ret, "mdd": mdd, "sharpe": sharpe,
            "sortino": sortino, "win_rate": wr, "trades": n_trades
        })
        tag = "✓" if mdd > -0.08 else " "
        if count % 50 == 0 or mdd > -0.08:
            print(f"[{count}/{total}]{tag} cb={cb:.2f} tir={tir:.2f} tv={tv:.2f} emerg={emerg} "
                  f"→ ret={ann_ret:.2%} mdd={mdd:.2%} sharpe={sharpe:.3f}")
    except Exception as e:
        print(f"[{count}/{total}] ERROR cb={cb:.2f} tir={tir:.2f} tv={tv:.2f} emerg={emerg}: {e}")

# ─── Analysis ───
df = pd.DataFrame(results)

print("\n" + "="*90)
print("ALL configurations with MDD > -8%:")
print("="*90)
valid = df[df["mdd"] > -0.08].sort_values("sharpe", ascending=False)
if valid.empty:
    print("None found! Showing best MDD configs:")
    valid = df.nsmallest(20, "mdd", keep="first").sort_values("sharpe", ascending=False)
    
for _, r in valid.head(30).iterrows():
    print(f"  cb={r.cb:.2f} tir={r.tir:.2f} tv={r.tv:.2f} emerg={r.emerg} "
          f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
          f"sortino={r.sortino:.3f} wr={r.win_rate:.1%} trades={r.trades:.0f}")

print("\n" + "="*90)
print("TOP-20 by Sharpe (MDD > -8%):")
print("="*90)
top_sharpe = df[df["mdd"] > -0.08].nlargest(20, "sharpe")
for _, r in top_sharpe.iterrows():
    print(f"  cb={r.cb:.2f} tir={r.tir:.2f} tv={r.tv:.2f} emerg={r.emerg} "
          f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
          f"sortino={r.sortino:.3f} wr={r.win_rate:.1%}")

print("\n" + "="*90)
print("TOP-20 by Returns (MDD > -8%):")
print("="*90)
top_ret = df[df["mdd"] > -0.08].nlargest(20, "ann_ret")
for _, r in top_ret.iterrows():
    print(f"  cb={r.cb:.2f} tir={r.tir:.2f} tv={r.tv:.2f} emerg={r.emerg} "
          f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
          f"sortino={r.sortino:.3f} wr={r.win_rate:.1%}")

# Pareto frontier: best Sharpe at each MDD bucket
print("\n" + "="*90)
print("Pareto Frontier (Sharpe vs MDD):")
print("="*90)
df["mdd_bucket"] = (df["mdd"] * 100).round(0)
pareto = df.loc[df.groupby("mdd_bucket")["sharpe"].idxmax()]
pareto = pareto.sort_values("mdd", ascending=False).head(25)
for _, r in pareto.iterrows():
    tag = "★" if r.mdd > -0.08 else " "
    print(f"  {tag} mdd={r.mdd:.2%} → cb={r.cb:.2f} tir={r.tir:.2f} tv={r.tv:.2f} emerg={r.emerg} "
          f"ret={r.ann_ret:.2%} sharpe={r.sharpe:.3f} sortino={r.sortino:.3f}")
