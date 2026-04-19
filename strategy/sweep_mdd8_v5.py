"""
sweep_mdd8_v5.py — 精细调优：围绕最佳配置 cb=0.05, rr=0.65, tv=0.15, cbt=3
加入TOP_N和RSI_OVERBOUGHT变量
"""
import sys, os, io, itertools, logging
import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
import risk_manager as rm
import signal_generator as sg
from data_fetcher import load_data
from feature_engine import build_features
from signal_generator import generate_monthly_holdings, get_rebalance_dates
from backtest_engine import (
    _build_execution_map, _get_holdings_list, _calc_portfolio_value,
    _execute_rebalance, _apply_trade
)
from risk_manager import calc_atr_weights, scale_to_target_vol, CircuitBreaker
from performance import annualized_return, max_drawdown, sharpe_ratio, monthly_win_rate, sortino_ratio

print("Loading data...")
close_matrix, ohlcv_panel = load_data(force_refresh=False)
close_matrix = close_matrix.loc[cfg.START_DATE:cfg.END_DATE]
ohlcv_panel = {sym: df.loc[cfg.START_DATE:cfg.END_DATE] for sym, df in ohlcv_panel.items()}
print("Building features...")
features = build_features(close_matrix, ohlcv_panel)

close = features["close"]
atr_df = features["atr"]
hist_vol_df = features["hist_vol"]
trading_dates = close.index

# Pre-generate monthly_holdings for different TOP_N and RSI combinations
holdings_cache = {}

def get_holdings(top_n, rsi_ob):
    key = (top_n, rsi_ob)
    if key not in holdings_cache:
        old_topn = cfg.TOP_N
        old_rsi = cfg.RSI_OVERBOUGHT
        old_sg_rsi = sg.RSI_OVERBOUGHT
        cfg.TOP_N = top_n
        cfg.RSI_OVERBOUGHT = rsi_ob
        sg.RSI_OVERBOUGHT = rsi_ob
        sg.TOP_N = top_n
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        h = generate_monthly_holdings(
            features["momentum_score"], features["close"], rsi=features.get("rsi")
        )
        sys.stdout = old_stdout
        
        cfg.TOP_N = old_topn
        cfg.RSI_OVERBOUGHT = old_rsi
        sg.RSI_OVERBOUGHT = old_rsi
        sg.TOP_N = old_topn
        
        holdings_cache[key] = h
    return holdings_cache[key]


def run_balanced(
    cb_trigger, risk_ratio, target_vol, cb_timeout,
    top_n, rsi_ob, initial_capital=1_000_000.0
):
    monthly_holdings = get_holdings(top_n, rsi_ob)
    signal_cols = [c for c in monthly_holdings.columns if not c.startswith("_")]
    rebal_signal_dates = monthly_holdings.index
    execute_map = _build_execution_map(rebal_signal_dates, trading_dates)
    pending_rebal = {}
    for sig_date, exec_date in execute_map.items():
        pending_rebal[exec_date] = sig_date
    
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
        
        # CB logic
        if scaler.is_triggered and scaler.trigger_date is not None:
            months = ((today.year - scaler.trigger_date.year) * 12 + (today.month - scaler.trigger_date.month))
            if months >= cb_timeout:
                scaler.is_triggered = False
                scaler.trigger_date = None
                scaler.peak_value = nav
        if not scaler.is_triggered and nav > (scaler.peak_value or 0):
            scaler.peak_value = nav
        if scaler.peak_value and scaler.peak_value > 0:
            dd = (nav - scaler.peak_value) / scaler.peak_value
        else:
            dd = 0.0
        if not scaler.is_triggered and dd <= -cb_trigger:
            scaler.is_triggered = True
            scaler.trigger_date = today
        if scaler.is_triggered and scaler.peak_value and nav >= scaler.peak_value * (1 - cb_trigger * 0.5):
            scaler.is_triggered = False
            scaler.trigger_date = None
            scaler.peak_value = nav
        
        is_scaled = scaler.is_triggered
        just_triggered = is_scaled and not was_triggered
        
        # Emergency rebalance
        if just_triggered and positions:
            n_def = len(cfg.DEFENSIVE_ETFS)
            emergency_weights = {sym: 0.98 / n_def for sym in cfg.DEFENSIVE_ETFS}
            trades = _execute_rebalance(
                today=today, target_weights=emergency_weights,
                current_positions=positions, close_row=close_row, nav=nav, cash=cash,
            )
            for t in trades:
                trade_records.append(t)
                cash, positions = _apply_trade(t, cash, positions)
            portfolio_value = _calc_portfolio_value(positions, close_row)
            nav = cash + portfolio_value
        
        # Monthly rebalance
        if today in pending_rebal:
            sig_date = pending_rebal[today]
            target_holdings = _get_holdings_list(monthly_holdings, sig_date, signal_cols)
            
            if is_scaled:
                n_def = len(cfg.DEFENSIVE_ETFS)
                target_weights = {sym: 0.98 / n_def for sym in cfg.DEFENSIVE_ETFS}
            else:
                old_tir = rm.TOTAL_INVEST_RATIO
                rm.TOTAL_INVEST_RATIO = risk_ratio
                risk_weights = calc_atr_weights(target_holdings, atr_row, close_row)
                risk_weights = scale_to_target_vol(risk_weights, hist_vol_row, target_vol=target_vol)
                rm.TOTAL_INVEST_RATIO = old_tir
                
                risk_total = sum(risk_weights.values())
                if risk_total > risk_ratio:
                    scale_f = risk_ratio / risk_total
                    risk_weights = {s: w * scale_f for s, w in risk_weights.items()}
                    risk_total = risk_ratio
                
                def_total = min(0.98 - risk_total, 0.98)
                if def_total < 0:
                    def_total = 0
                
                n_def = len(cfg.DEFENSIVE_ETFS)
                target_weights = dict(risk_weights)
                for ds in cfg.DEFENSIVE_ETFS:
                    target_weights[ds] = target_weights.get(ds, 0) + def_total / n_def
            
            trades = _execute_rebalance(
                today=today, target_weights=target_weights,
                current_positions=positions, close_row=close_row, nav=nav, cash=cash,
            )
            for t in trades:
                trade_records.append(t)
                cash, positions = _apply_trade(t, cash, positions)
            portfolio_value = _calc_portfolio_value(positions, close_row)
            nav = cash + portfolio_value
        
        nav_records.append({"date": today, "nav": nav})
    
    nav_df = pd.DataFrame(nav_records).set_index("date")
    nav_s = nav_df["nav"]
    
    return (annualized_return(nav_s), max_drawdown(nav_s)[0], sharpe_ratio(nav_s), 
            sortino_ratio(nav_s), monthly_win_rate(nav_s)[0], len(trade_records))


# ─── Fine-grained sweep ───
# Around the best: cb=0.05, rr=0.65, tv=0.15, cbt=3, top_n=4, rsi=86
cb_triggers = [0.04, 0.045, 0.048, 0.050, 0.052, 0.055, 0.06]
risk_ratios = [0.55, 0.58, 0.60, 0.63, 0.65, 0.67, 0.70, 0.75]
target_vols = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.20]
cb_timeouts = [2, 3, 4]
top_ns = [3, 4, 5]
rsi_obs = [80, 83, 86, 89, 0]

total = len(cb_triggers) * len(risk_ratios) * len(target_vols) * len(cb_timeouts) * len(top_ns) * len(rsi_obs)
print(f"Fine-grained sweep: {total} configurations...\n")

# First pass: sweep cb × rr × tv × cbt with default top_n=4, rsi=86
print("Phase 1: Core params (top_n=4, rsi=86)...")
results1 = []
count = 0
total1 = len(cb_triggers) * len(risk_ratios) * len(target_vols) * len(cb_timeouts)
for cb, rr, tv, cbt in itertools.product(cb_triggers, risk_ratios, target_vols, cb_timeouts):
    count += 1
    try:
        ann_ret, mdd, sharpe, sortino, wr, nt = run_balanced(
            cb, rr, tv, cbt, top_n=4, rsi_ob=86
        )
        results1.append({
            "cb": cb, "rr": rr, "tv": tv, "cbt": cbt, "topn": 4, "rsi": 86,
            "ann_ret": ann_ret, "mdd": mdd, "sharpe": sharpe, "sortino": sortino,
            "win_rate": wr, "trades": nt
        })
    except Exception as e:
        pass
    if count % 200 == 0:
        print(f"  [{count}/{total1}]")

df1 = pd.DataFrame(results1)
valid1 = df1[df1["mdd"] > -0.08]
print(f"\n  Phase 1 done: {len(valid1)} configs with MDD > -8%")

# Get top-10 core configs
top_core = valid1.nlargest(10, "sharpe")
print("\n  Top-10 core configs:")
for _, r in top_core.iterrows():
    print(f"    cb={r.cb:.3f} rr={r.rr:.2f} tv={r.tv:.2f} cbt={r.cbt:.0f} "
          f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f}")

# Phase 2: For top core configs, sweep TOP_N and RSI
print(f"\nPhase 2: TOP_N × RSI sweep on top configs...")
results2 = []

for _, core in top_core.iterrows():
    for topn, rsi in itertools.product(top_ns, rsi_obs):
        if topn == 4 and rsi == 86:
            continue  # already computed
        try:
            ann_ret, mdd, sharpe, sortino, wr, nt = run_balanced(
                core.cb, core.rr, core.tv, int(core.cbt), top_n=topn, rsi_ob=rsi
            )
            results2.append({
                "cb": core.cb, "rr": core.rr, "tv": core.tv, "cbt": core.cbt,
                "topn": topn, "rsi": rsi,
                "ann_ret": ann_ret, "mdd": mdd, "sharpe": sharpe, "sortino": sortino,
                "win_rate": wr, "trades": nt
            })
        except:
            pass

df_all = pd.concat([df1, pd.DataFrame(results2)], ignore_index=True) if results2 else df1
valid_all = df_all[df_all["mdd"] > -0.08].copy()

print(f"\n{'='*110}")
print(f"FINAL RESULTS: {len(valid_all)} configs with MDD > -8%")
print(f"{'='*110}")

print(f"\nTOP-30 by Sharpe (MDD > -8%):")
print("-"*110)
top = valid_all.nlargest(30, "sharpe")
for _, r in top.iterrows():
    print(f"  cb={r.cb:.3f} rr={r.rr:.2f} tv={r.tv:.2f} cbt={r.cbt:.0f} topn={r.topn:.0f} rsi={r.rsi:.0f} "
          f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
          f"sortino={r.sortino:.3f} wr={r.win_rate:.1%} tr={r.trades:.0f}")

print(f"\nTOP-20 by Returns (MDD > -8%, Sharpe > 0.7):")
print("-"*110)
valid_high = valid_all[valid_all["sharpe"] > 0.7]
for _, r in valid_high.nlargest(20, "ann_ret").iterrows():
    print(f"  cb={r.cb:.3f} rr={r.rr:.2f} tv={r.tv:.2f} cbt={r.cbt:.0f} topn={r.topn:.0f} rsi={r.rsi:.0f} "
          f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
          f"sortino={r.sortino:.3f} wr={r.win_rate:.1%}")

# Overall best by composite score
valid_all["score"] = valid_all["sharpe"] * 2 + valid_all["ann_ret"] * 10 + (valid_all["mdd"] + 0.08) * 5
print(f"\nTOP-20 OVERALL (composite score = 2×Sharpe + 10×Return + 5×(MDD_margin)):")
print("-"*110)
for _, r in valid_all.nlargest(20, "score").iterrows():
    print(f"  cb={r.cb:.3f} rr={r.rr:.2f} tv={r.tv:.2f} cbt={r.cbt:.0f} topn={r.topn:.0f} rsi={r.rsi:.0f} "
          f"→ ret={r.ann_ret:.2%} mdd={r.mdd:.2%} sharpe={r.sharpe:.3f} "
          f"sortino={r.sortino:.3f} wr={r.win_rate:.1%}")
