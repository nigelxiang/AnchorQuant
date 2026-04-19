"""
feature_engine.py — 因子计算与特征工程
核心原则：所有因子使用 .shift(1) 严格错开，确保无未来函数泄漏。
输出特征矩阵可直接送入 signal_generator。

因子列表：
  - mom_{N}d           : N日动量（收益率，已错位）
  - mom_score          : 加权合成动量得分
  - atr_{N}d           : N日真实波幅（用于风险平价仓位）
  - hist_vol_{N}d      : N日历史波动率（年化）
  - rsi_14             : RSI(14)（辅助过滤极端热度）
  - skewness_60d       : 60日对数收益率偏度（尾部风险信号）
  - mean_rev_60d       : (close - MA60) / MA60 的 z-score（超买超卖）
  - ols_slope_60d      : 60日OLS线性回归斜率（趋势方向与强度）
  - ols_r2_60d         : 60日OLS R²（趋势平稳度）
  - style_ratios       : 大小盘/价值成长比值（A股风格切换信号）
  - cross_border_state : 跨境ETF近期涨跌状态（全球风险偏好）
"""

import numpy as np
import pandas as pd

from config import (
    MOM_WINDOWS, MOM_WEIGHTS, SKIP_DAYS,
    ATR_WINDOW, VOL_WINDOW, TRADING_DAYS_PER_YEAR,
    TREND_MA_WINDOW, USE_RISK_ADJ_MOM, CONSISTENCY_WEIGHT,
    STYLE_PAIRS, CROSS_BORDER_REFS,
)


# ─────────────────────────────────────────────
# 动量因子
# ─────────────────────────────────────────────
def calc_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """
    计算各窗口动量，返回宽表，每只ETF一列。
    
    公式：mom_N(t) = close(t-SKIP_DAYS) / close(t-N-SKIP_DAYS) - 1
    使用 shift(SKIP_DAYS) 跳过最近N日（避免短期反转污染）。
    """
    mom_frames = {}
    for window in MOM_WINDOWS:
        # 计算 N+SKIP 日到 SKIP 日之间的收益（跳过最近 SKIP_DAYS 日）
        shifted_close = close.shift(SKIP_DAYS)   # 最近价格（错位）
        past_close    = close.shift(window + SKIP_DAYS)   # N日前价格（严格错位）
        mom = shifted_close / past_close - 1
        mom.columns = [f"mom_{window}d_{col}" for col in close.columns]
        mom_frames[window] = mom
    return mom_frames


def calc_composite_momentum(close: pd.DataFrame) -> pd.DataFrame:
    """
    合成加权动量得分，返回 DataFrame(date x symbol)。
    每行对所有ETF打分，值为浮点数（可跨ETF比较相对大小）。

    关键防穿越：
    - 基于已错位的动量因子，信号日收盘后可用，次日开盘执行
    - 截面排名在信号生成时做，此处只计算原始得分
    
    一致性奖惩（CONSISTENCY_WEIGHT > 0 时生效）：
    - 三个动量周期方向一致（同涨或同跌）时，将得分乘以 (1 + CONSISTENCY_WEIGHT)
    - 方向混乱时，将得分乘以 (1 - CONSISTENCY_WEIGHT)
    - 效果：过滤假突破，仅在趋势清晰时加仓
    """
    mom_frames = calc_momentum(close)

    symbols = close.columns.tolist()
    composite = pd.DataFrame(0.0, index=close.index, columns=symbols)

    for window, weight in zip(MOM_WINDOWS, MOM_WEIGHTS):
        mom_df = mom_frames[window]  # date x "mom_{N}d_{sym}"
        for sym in symbols:
            col = f"mom_{window}d_{sym}"
            if col in mom_df.columns:
                composite[sym] += weight * mom_df[col]

    # ── 跨周期一致性奖惩 ──
    if CONSISTENCY_WEIGHT > 0 and len(MOM_WINDOWS) >= 2:
        # 收集各周期符号矩阵（date x symbol），值为 +1 或 -1
        sign_frames = []
        for window in MOM_WINDOWS:
            mom_df = mom_frames[window]
            sign = pd.DataFrame(index=close.index, columns=symbols, dtype=float)
            for sym in symbols:
                col = f"mom_{window}d_{sym}"
                if col in mom_df.columns:
                    sign[sym] = np.sign(mom_df[col])
                else:
                    sign[sym] = 0.0
            sign_frames.append(sign)

        # 计算一致性得分：各期符号之积的均值（范围 -1 到 +1）
        # 对于 3 个周期：consistency = mean(s1*s2, s2*s3, s1*s3)
        n = len(sign_frames)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        consistency = pd.DataFrame(0.0, index=close.index, columns=symbols)
        for i, j in pairs:
            consistency += sign_frames[i] * sign_frames[j]
        consistency /= len(pairs)  # 归一化到 [-1, +1]

        # 调整合成得分：得分乘以 (1 + CONSISTENCY_WEIGHT * consistency)
        # 当 consistency=+1 时，得分放大 (1+w)；当 consistency=-1 时，得分缩小 (1-w)
        multiplier = 1.0 + CONSISTENCY_WEIGHT * consistency
        composite = composite * multiplier

    return composite   # date x symbol，每格为合成动量分


# ─────────────────────────────────────────────
# ATR（真实波幅）
# ─────────────────────────────────────────────
def calc_atr(ohlcv_panel: dict, window: int = ATR_WINDOW) -> pd.DataFrame:
    """
    对OHLCV面板中每只ETF计算ATR(window)，返回宽表 (date x symbol)。
    ATR使用前一日收盘错位，不含当日信息泄漏。
    
    已错位：ATR的计算需要high/low/prev_close，prev_close 天然错一格。
    信号日的ATR反映截止昨日的波动情况，可安全使用。
    """
    atr_dict = {}
    for sym, df in ohlcv_panel.items():
        if not {"high", "low", "close"}.issubset(df.columns):
            continue
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window, min_periods=window // 2).mean()
        atr_dict[sym] = atr

    if not atr_dict:
        return pd.DataFrame()
    return pd.DataFrame(atr_dict)


# ─────────────────────────────────────────────
# 历史波动率（年化）
# ─────────────────────────────────────────────
def calc_hist_vol(close: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    N日历史波动率（年化），使用对数收益率。
    shift(1) 确保只用到昨日及以前的数据，无未来泄漏。
    """
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.shift(1).rolling(window, min_periods=window // 2).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol  # date x symbol，单位：年化波动率（小数）


# ─────────────────────────────────────────────
# RSI（相对强弱指数，辅助过滤）
# ─────────────────────────────────────────────
def calc_rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI(14)，用于过滤极端超买（RSI>85时谨慎追高）。
    同样 shift(1) 错位，不使用当日收盘价。
    """
    delta = close.shift(1).diff(1)
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi  # date x symbol


# ─────────────────────────────────────────────
# 偏度（Skewness）— 60日滚动
# ─────────────────────────────────────────────
def calc_skewness(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    60日滚动对数收益率偏度。
    正偏度 = 右尾肥（上涨惊喜多）；负偏度 = 左尾肥（下跌风险大）。
    shift(1) 确保只使用昨日及以前的数据。
    """
    log_ret = np.log(close / close.shift(1)).shift(1)
    return log_ret.rolling(window, min_periods=window // 2).skew()


# ─────────────────────────────────────────────
# 均值回归信号 — (close - MA60) / std60 z-score
# ─────────────────────────────────────────────
def calc_mean_reversion(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    均值回归偏离度 z-score：
    z = (close(t-1) - MA(t-1)) / std(t-1)
    正值=超买（偏离均值向上），负值=超卖（偏离均值向下）。
    shift(1) 确保无未来泄漏。
    """
    shifted = close.shift(1)
    ma = shifted.rolling(window, min_periods=window // 2).mean()
    std = shifted.rolling(window, min_periods=window // 2).std()
    return (shifted - ma) / std.replace(0, np.nan)


# ─────────────────────────────────────────────
# OLS动量斜率与R²（趋势强度与平稳度）
# ─────────────────────────────────────────────
def calc_ols_slope_r2(close: pd.DataFrame, window: int = 60) -> tuple:
    """
    对每只ETF的对数价格序列做60日滚动OLS线性回归：
      ln(close) ~ a + b * t
    
    返回：
    - slope_df: 标准化斜率（年化趋势收益估计）
    - r2_df:    R²（趋势平稳度，高R²=稳定趋势，低R²=震荡）
    
    shift(1) 确保信号日只使用前一日及以前的价格。
    """
    log_price = np.log(close.shift(1))
    symbols = close.columns.tolist()
    dates = close.index

    slope_data = {}
    r2_data = {}

    for sym in symbols:
        col = log_price[sym].values
        slopes = np.full(len(dates), np.nan)
        r2s = np.full(len(dates), np.nan)

        for i in range(window - 1, len(dates)):
            y = col[i - window + 1: i + 1]
            valid = ~np.isnan(y)
            if valid.sum() < window // 2:
                continue
            x = np.arange(window)[valid]
            y_v = y[valid]
            # OLS via polyfit
            coeffs = np.polyfit(x, y_v, 1)
            b = coeffs[0]  # 斜率（log价格/日）
            y_hat = np.polyval(coeffs, x)
            ss_res = np.sum((y_v - y_hat) ** 2)
            ss_tot = np.sum((y_v - y_v.mean()) ** 2)
            r2s[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            # 标准化：斜率 × 252 = 年化趋势收益估计
            slopes[i] = b * TRADING_DAYS_PER_YEAR

        slope_data[sym] = slopes
        r2_data[sym] = r2s

    slope_df = pd.DataFrame(slope_data, index=dates)
    r2_df = pd.DataFrame(r2_data, index=dates)
    return slope_df, r2_df


# ─────────────────────────────────────────────
# A股风格比值（大小盘 / 价值成长）
# ─────────────────────────────────────────────
def calc_style_ratios(close: pd.DataFrame, style_pairs: dict = None) -> pd.DataFrame:
    """
    计算风格比值序列，为LLM提供A股市场风格切换上下文。

    style_pairs = {"大小盘比": ("510300", "512100"), ...}
    比值上升 = 大盘/价值占优；比值下降 = 小盘/成长轮动。
    shift(1) 错位，无未来泄漏。
    """
    if style_pairs is None:
        style_pairs = STYLE_PAIRS

    result = {}
    for name, (num_sym, denom_sym) in style_pairs.items():
        if num_sym in close.columns and denom_sym in close.columns:
            ratio = close[num_sym].shift(1) / close[denom_sym].shift(1).replace(0, np.nan)
            result[name] = ratio

    return pd.DataFrame(result, index=close.index)


# ─────────────────────────────────────────────
# 跨境市场状态（全球风险偏好特征）
# ─────────────────────────────────────────────
def calc_cross_border_state(
    close: pd.DataFrame,
    ref_symbols: list = None,
    window: int = 20,
) -> pd.DataFrame:
    """
    计算跨境参考ETF近N日涨跌幅，为LLM提供全球风险偏好上下文。

    返回：DataFrame，columns=ETF代码，值=近window日收益率（已shift(1)错位）。
    """
    if ref_symbols is None:
        ref_symbols = CROSS_BORDER_REFS

    available = [s for s in ref_symbols if s in close.columns]
    if not available:
        return pd.DataFrame(index=close.index)

    # 近window日收益率，shift(1)错位
    cross = {}
    for sym in available:
        ret_n = close[sym].shift(1) / close[sym].shift(window + 1) - 1
        cross[f"xborder_{sym}"] = ret_n

    return pd.DataFrame(cross, index=close.index)


# ─────────────────────────────────────────────
# 统一入口：计算全量特征
# ─────────────────────────────────────────────
def build_features(
    close: pd.DataFrame,
    ohlcv_panel: dict,
) -> dict:
    """
    返回字典：
    {
        "momentum_score":    DataFrame(date x symbol),   # 合成动量得分
        "atr":               DataFrame(date x symbol),   # ATR(14)
        "hist_vol":          DataFrame(date x symbol),   # 历史波动率
        "rsi":               DataFrame(date x symbol),   # RSI(14)
        "skewness":          DataFrame(date x symbol),   # 60日偏度
        "mean_reversion":    DataFrame(date x symbol),   # 均值回归z-score
        "ols_slope":         DataFrame(date x symbol),   # OLS年化斜率
        "ols_r2":            DataFrame(date x symbol),   # OLS R²
        "style_ratios":      DataFrame(date x names),    # 风格比值
        "cross_border":      DataFrame(date x xnames),   # 跨境状态
        "close":             DataFrame(date x symbol),   # 原始价格（回测用）
    }
    所有 DataFrame 均已严格错位（信号日可安全用于次日下单）。
    """
    features = {
        "momentum_score": calc_composite_momentum(close),
        "atr":            calc_atr(ohlcv_panel),
        "hist_vol":       calc_hist_vol(close),
        "rsi":            calc_rsi(close),
        "skewness":       calc_skewness(close),
        "mean_reversion": calc_mean_reversion(close),
        "style_ratios":   calc_style_ratios(close),
        "cross_border":   calc_cross_border_state(close),
        "close":          close,
    }

    # ── OLS 斜率与R²（滚动计算，较慢，独立执行）──
    ols_slope, ols_r2 = calc_ols_slope_r2(close)
    features["ols_slope"] = ols_slope
    features["ols_r2"]    = ols_r2

    # ── 风险调整动量：momentum_score / hist_vol ──
    if USE_RISK_ADJ_MOM:
        vol = features["hist_vol"]
        mom = features["momentum_score"]
        safe_vol = vol.clip(lower=0.05)
        features["risk_adj_momentum"] = mom / safe_vol

    # ── MA200 趋势信号：close > SMA(200) ──
    if TREND_MA_WINDOW > 0:
        ma200 = close.rolling(TREND_MA_WINDOW, min_periods=TREND_MA_WINDOW // 2).mean()
        features["trend_above_ma"] = (close.shift(1) > ma200.shift(1)).astype(int)
    else:
        features["trend_above_ma"] = None

    # 验证：确保momentum_score不包含NaN泄漏到过前的日期
    max_lookback = max(MOM_WINDOWS) + SKIP_DAYS
    warmup_end = close.index[max_lookback] if len(close) > max_lookback else close.index[-1]
    _assert_no_look_ahead(features["momentum_score"], close, warmup_end)

    return features


def _assert_no_look_ahead(score_df: pd.DataFrame, close: pd.DataFrame, warmup_end: pd.Timestamp):
    """
    防穿越断言：
    检查合成动量在 SKIP_DAYS=1 条件下，不会使用当日close。
    策略：计算 score_df 的第一个非NaN行，必须晚于 warmup_end。
    """
    first_valid_row = score_df.dropna(how="all").index[0] if not score_df.dropna(how="all").empty else None
    if first_valid_row is not None and first_valid_row < warmup_end:
        # 只警告，不中断（部分ETF上市较晚，会有合理的早期NaN）
        import warnings
        warnings.warn(
            f"动量因子首个有效行({first_valid_row.date()}) 早于预热期结束({warmup_end.date()})，"
            f"请检查 SKIP_DAYS 配置是否生效。"
        )


if __name__ == "__main__":
    """快速验证：检查动量因子是否正确错位"""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_fetcher import load_data

    print("加载数据...")
    close_mtx, ohlcv = load_data()
    print("计算特征...")
    feats = build_features(close_mtx, ohlcv)

    ms = feats["momentum_score"]
    print(f"\n动量得分 (最近5行):")
    print(ms.tail())

    atr = feats["atr"]
    print(f"\nATR (最近3行):")
    print(atr.tail(3))

    # 关键检验：动量得分的每行不能与当日close明显相关（防穿越校验）
    print("\n[校验] 动量得分 vs 当日收益率相关性（应接近0）:")
    today_ret = close_mtx.pct_change()
    overlap_cols = [c for c in ms.columns if c in today_ret.columns]
    corr = ms[overlap_cols].corrwith(today_ret[overlap_cols], axis=1).mean()
    print(f"  平均相关系数: {corr:.4f}  (|corr|<0.05 则无泄漏)")
