"""
signal_generator.py — 双动量信号生成
策略逻辑：Gary Antonacci 双动量（Dual Momentum）本土化版本，支持 hybrid LLM模式

流程（月末每月执行一次）：
1. 相对动量：对ETF池按合成动量得分截面排名，取 Top-N
2. 绝对动量：Top-N中，若某ETF的动量得分 低于 防御资产最高得分，则替换为防御资产
3. [hybrid模式] 调用 DeepSeek 双Agent，用LLM权重覆盖量化排名
4. 输出目标仓位字典 {date: {symbol: weight}}，权重由 risk_manager 填充

注意：信号基于月末收盘价计算，次月第一个交易日开盘执行（T+1语义延伸到月度）。
"""

import numpy as np
import pandas as pd
import logging

import config as _cfg
from config import DEFENSIVE_ETFS
from risk_manager import compute_position_weights, apply_static_defensive_allocation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 月度调仓日历生成
# ─────────────────────────────────────────────
def get_rebalance_dates(date_index: pd.DatetimeIndex, freq: str = None) -> pd.DatetimeIndex:
    """
    从交易日历中提取调仓日期。
    freq=None 时从config.REBAL_FREQ读取（运行时），避免导入时固化。
    支持: "ME"(月末), "SME"(半月末), "2W"(双周), "W"(周)
    """
    if freq is None:
        freq = _cfg.REBAL_FREQ
    # 用交易日序号作为值，resample取last得到每周期最后一个交易日序号
    dummy = pd.Series(range(len(date_index)), index=date_index)
    rebal = dummy.resample(freq).last().dropna().astype(int)
    valid_dates = date_index[rebal.values]
    return valid_dates


# ─────────────────────────────────────────────
# 相对动量截面排名
# ─────────────────────────────────────────────
def rank_by_momentum(
    score_row: pd.Series,
    top_n: int = None,
    exclude_symbols: list = None,
    prev_holdings: list = None,
) -> list:
    """
    对单行动量得分做截面排名，返回得分最高的 top_n 只ETF列表。
    - 排除防御资产参与排名（防御资产只在绝对动量判断时使用）
    - 排除 exclude_symbols 中的标的（下跌趋势、停牌等）
    - 持仓缓冲：prev_holdings 中的现有持仓若排名在 top_n+HOLD_BUFFER 内则保留
    - 返回 [] 若可用标的不足
    """
    if top_n is None:
        top_n = _cfg.TOP_N

    exclude = set(DEFENSIVE_ETFS)
    if exclude_symbols:
        exclude.update(exclude_symbols)

    scores = score_row.drop(labels=[s for s in exclude if s in score_row.index], errors="ignore")
    scores = scores.dropna()

    if scores.empty:
        return []

    # 持仓缓冲逻辑：现有持仓如果还在 top_n+HOLD_BUFFER 内，优先保留
    if prev_holdings:
        extended_n = top_n + _cfg.HOLD_BUFFER
        wider_top = scores.nlargest(min(extended_n, len(scores))).index.tolist()
        # 保留现有持仓中仍在放宽范围内的
        kept = [s for s in prev_holdings if s in wider_top and s not in exclude]
        # 新增的席位数
        slots_left = top_n - len(kept)
        if slots_left > 0:
            # 从排名靠前的非现有持仓中选
            new_candidates = [s for s in scores.nlargest(len(scores)).index if s not in kept]
            kept.extend(new_candidates[:slots_left])
        return kept[:top_n]

    top = scores.nlargest(top_n).index.tolist()
    return top


# ─────────────────────────────────────────────
# 绝对动量过滤（核心防御机制）
# ─────────────────────────────────────────────
def apply_absolute_momentum(
    candidates: list,
    score_row: pd.Series,
) -> list:
    """
    绝对动量过滤：
    - 若候选ETF的动量得分 <= 防御资产中最大得分，则替换为防御资产
    - 防御资产得分代表"持有债券的机会成本基准线"
    
    返回过滤后的持仓列表（可能包含防御替代项）。
    """
    # 某些子策略（如进攻引擎）会显式剔除防御资产，此时无法构造绝对动量基准线，
    # 应直接跳过过滤，而不是对每个调仓日重复报警。
    if not any(def_sym in score_row.index for def_sym in DEFENSIVE_ETFS):
        return candidates

    # 防御资产基准分（取防御ETF池中最大得分）
    defensive_scores = []
    for def_sym in DEFENSIVE_ETFS:
        if def_sym in score_row.index and not pd.isna(score_row[def_sym]):
            defensive_scores.append(score_row[def_sym])

    # 若防御资产无数据，降级处理：不做绝对动量过滤
    if not defensive_scores:
        logger.warning("防御资产无动量数据，跳过绝对动量过滤")
        return candidates

    defensive_threshold = max(defensive_scores)

    result = []
    defensive_added = False
    for sym in candidates:
        sym_score = score_row.get(sym, np.nan)
        # 双重过滤：1) 弱于防御资产  2) 得分不存在
        if pd.isna(sym_score) or sym_score <= defensive_threshold:
            # 当前 ETF 动量弱于防御资产 → 替换为最优防御ETF
            if not defensive_added:
                best_def = max(
                    (s for s in DEFENSIVE_ETFS if s in score_row.index and not pd.isna(score_row[s])),
                    key=lambda s: score_row[s],
                    default=DEFENSIVE_ETFS[0],
                )
                result.append(best_def)
                defensive_added = True
            # 不添加弱势ETF
        else:
            result.append(sym)

    # 若全部替换为防御且防御资产也只添加了一次，补足至 top_n 数量用同一防御资产（等权）
    return result if result else [DEFENSIVE_ETFS[0]]


# ─────────────────────────────────────────────
# 相关性感知去重（核心分散化机制）
# ─────────────────────────────────────────────
def apply_correlation_filter(
    ranked_pool: list,
    close_window: pd.DataFrame,
    top_n: int = None,
    max_corr: float = None,
    score_row: pd.Series = None,
) -> list:
    """
    从按动量排名的候选池中，贪心选取 top_n 只相关性低的ETF。

    算法：
    1. 候选池已按动量降序排列
    2. 逐一遍历候选，若与已选持仓的相关性 < max_corr 则加入
    3. 若候选池不足，以原排名顺序补足

    经济学依据：持仓间高相关 → 分散化失效 → 组合实际波动 ≈ 单etf波动
    → 相同期望收益下夏普比率下降。强制低相关可实现真实分散。

    注意：仅在有足够历史数据时生效（窗口不足则退化为原始排名）。
    """
    if top_n is None:
        top_n = _cfg.TOP_N
    if max_corr is None:
        max_corr = _cfg.MAX_PAIR_CORR

    if len(ranked_pool) <= 1 or close_window is None or close_window.empty:
        return ranked_pool[:top_n]

    # 只取可计算相关性的ETF（在close_window中有列）
    available = [s for s in ranked_pool if s in close_window.columns]
    if len(available) <= 1:
        return ranked_pool[:top_n]

    # 计算对数收益率相关矩阵
    log_ret = np.log(close_window[available] / close_window[available].shift(1)).dropna()
    corr_window = _cfg.CORR_FILTER_WINDOW
    if len(log_ret) < max(20, corr_window // 3):
        # 数据不足，跳过过滤
        return ranked_pool[:top_n]

    corr_matrix = log_ret.tail(corr_window).corr()

    # 贪心选取：按动量排名逐一检查是否与已选集中的相关性过高
    selected = []
    for sym in ranked_pool:
        if len(selected) >= top_n:
            break
        if sym not in corr_matrix.index:
            # 没有相关性数据，直接加入（保险处理）
            selected.append(sym)
            continue
        if not selected:
            selected.append(sym)
            continue
        # 检查与已选的最大相关性
        max_c = max(
            corr_matrix.loc[sym, s]
            for s in selected
            if s in corr_matrix.columns
        )
        if max_c < max_corr:
            selected.append(sym)
        # else: 跳过此ETF，与已选过于相关

    # 若选不够 top_n（相关性全超标），用原排名补足
    if len(selected) < top_n:
        for sym in ranked_pool:
            if sym not in selected:
                selected.append(sym)
            if len(selected) >= top_n:
                break

    return selected


# ─────────────────────────────────────────────
# 生成月度目标持仓（无权重，权重由 risk_manager 计算）
# ─────────────────────────────────────────────
def _generate_single_engine_holdings(
    momentum_score: pd.DataFrame,
    close: pd.DataFrame,
    risk_adj_momentum: pd.DataFrame = None,
    trend_above_ma: pd.DataFrame = None,
    rsi: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    按月末调仓日期生成目标持仓列表。
    
    增强功能：
    - risk_adj_momentum: 风险调整动量（momentum/vol），用于排名替代原始动量
    - trend_above_ma:    趋势过滤信号，1=上升趋势，0=下降趋势
    
    持仓缓冲：现有持仓在 TOP_N+HOLD_BUFFER 排名内不换出。
    返回：DataFrame，index=调仓信号日，columns=ETF代码，值=1（持有）或0
    """
    # 使用风险调整动量排名（如果可用且开关打开）
    ranking_score = risk_adj_momentum if (_cfg.USE_RISK_ADJ_MOM and risk_adj_momentum is not None) else momentum_score

    rebal_dates = get_rebalance_dates(momentum_score.index)
    all_symbols = momentum_score.columns.tolist()

    holdings_records = []
    prev_holdings = []   # 上月持仓列表（用于持仓缓冲）

    for date in rebal_dates:
        if date not in ranking_score.index:
            continue

        score_row = ranking_score.loc[date]

        # 1. 找到有效（非NaN）的ETF子集
        valid_mask = score_row.notna()
        # 同时检查 close 是否有数据（排除停牌超5日的ETF）
        if date in close.index:
            close_row = close.loc[date]
            valid_mask &= close_row.notna()

        # RSI过热过滤：RSI超过阈值的ETF从排名中剔除
        if _cfg.RSI_OVERBOUGHT > 0 and rsi is not None and date in rsi.index:
            rsi_row = rsi.loc[date]
            for sym in all_symbols:
                if sym not in DEFENSIVE_ETFS and sym in rsi_row.index:
                    if not pd.isna(rsi_row[sym]) and rsi_row[sym] > _cfg.RSI_OVERBOUGHT:
                        valid_mask[sym] = False

        # 2. 趋势软过滤：下跌趋势ETF得分加减法惩罚
        #    上升趋势(MA200以上): 不变
        #    下跌趋势(MA200以下): score -= 0.03 (绝对值惩罚，正确处理正负分)
        if trend_above_ma is not None and date in trend_above_ma.index:
            trend_row = trend_above_ma.loc[date]
            score_row = score_row.copy()
            for sym in all_symbols:
                if sym not in DEFENSIVE_ETFS and sym in trend_row.index and sym in score_row.index:
                    if trend_row.get(sym, 1) == 0:
                        score_row[sym] -= 0.03

        available = score_row[valid_mask].index.tolist()
        unavailable = [s for s in all_symbols if s not in available]

        # 3. 相对动量排名：取扩大候选池（2*TOP_N），留给相关性过滤足够空间
        extended_pool = rank_by_momentum(
            score_row[valid_mask],
            top_n=_cfg.TOP_N * 2,
            exclude_symbols=unavailable,
            prev_holdings=prev_holdings,
        )

        # 3b. 相关性感知去重：从扩大候选池中选出低相关的 TOP_N
        close_window = close.loc[:date].tail(_cfg.CORR_FILTER_WINDOW + 5)
        orig_score_for_corr = momentum_score.loc[date] if date in momentum_score.index else score_row
        candidates = apply_correlation_filter(
            extended_pool,
            close_window,
            top_n=_cfg.TOP_N,
            score_row=orig_score_for_corr,
        )

        # 4. 绝对动量过滤（使用原始动量得分判断，不用风险调整分）
        orig_score_row = momentum_score.loc[date] if date in momentum_score.index else score_row
        final_holdings = apply_absolute_momentum(candidates, orig_score_row)

        # 记录为二值宽表行
        row = {sym: 0 for sym in all_symbols}
        for sym in final_holdings:
            if sym in row:
                row[sym] = 1
        row["_date"] = date
        row["_n_holdings"] = len(set(final_holdings))
        holdings_records.append(row)

        # 更新上月持仓
        prev_holdings = [s for s in final_holdings if s not in DEFENSIVE_ETFS]

    result = pd.DataFrame(holdings_records).set_index("_date")
    result.index.name = "date"

    logger.info(
        f"信号生成完成: {len(result)}个调仓日, "
        f"平均持仓数量={result['_n_holdings'].mean():.1f}"
    )
    return result


def _apply_dual_engine_defensive_hold(
    holdings: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """
    双引擎防守连续持有保护：
    - 进入防守引擎后，可额外连续持有若干个调仓周期
    - 或要求基准重新站稳长期均线缓冲后，才允许切回进攻
    """
    min_hold_rebals = getattr(_cfg, "DUAL_DEFENSIVE_HOLD_MIN_REBALS", 0)
    reentry_buffer = getattr(_cfg, "DUAL_DEFENSIVE_REENTRY_BUFFER", 0.0)

    if "_engine" not in holdings.columns:
        return holdings
    if min_hold_rebals <= 0 and reentry_buffer <= 0:
        return holdings

    benchmark = _cfg.DUAL_SWITCH_BENCHMARK
    ma_window = _cfg.DUAL_SWITCH_MA_WINDOW
    benchmark_ma = None
    if benchmark in close.columns and ma_window > 0:
        benchmark_ma = close[benchmark].shift(1).rolling(
            ma_window,
            min_periods=max(20, ma_window // 2),
        ).mean()

    signal_cols = [c for c in holdings.columns if not c.startswith("_")]
    protected = holdings.copy()
    last_defensive_row = None
    rebals_left = 0

    for date in protected.index:
        current_engine = str(protected.loc[date, "_engine"])
        if current_engine == "defensive":
            last_defensive_row = protected.loc[date].copy()
            rebals_left = min_hold_rebals
            continue

        below_reentry_line = False
        if benchmark_ma is not None and date in benchmark_ma.index and pd.notna(benchmark_ma.loc[date]):
            below_reentry_line = close.loc[date, benchmark] < benchmark_ma.loc[date] * (1 + reentry_buffer)

        if last_defensive_row is not None and (rebals_left > 0 or below_reentry_line):
            for sym in signal_cols:
                protected.loc[date, sym] = last_defensive_row.get(sym, 0)
            protected.loc[date, "_n_holdings"] = last_defensive_row.get("_n_holdings", 0)
            protected.loc[date, "_engine"] = "defensive_hold"
            protected.loc[date, "_switch_reason"] = "defensive_hysteresis"
            if rebals_left > 0:
                rebals_left -= 1

    return protected


def generate_monthly_holdings_dual_engine(
    momentum_score: pd.DataFrame,
    close: pd.DataFrame,
    risk_adj_momentum: pd.DataFrame = None,
    trend_above_ma: pd.DataFrame = None,
    rsi: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    双引擎量化信号：
    1. 进攻子策略：在非防御ETF池中做原有双动量选股
    2. 防守子策略：在防御ETF+黄金小池中选择最优避险资产
    3. 月度切换层：若宽基跌破长期均线，或防守引擎代表资产相对更强，则切到防守引擎
    """
    all_symbols = close.columns.tolist()
    defensive_pool = [sym for sym in _cfg.DUAL_ENGINE_DEFENSIVE_POOL if sym in all_symbols]
    offensive_pool = [sym for sym in all_symbols if sym not in DEFENSIVE_ETFS]

    if not offensive_pool or not defensive_pool:
        logger.warning("双引擎所需ETF池不完整，回退到单引擎模式")
        return _generate_single_engine_holdings(
            momentum_score=momentum_score,
            close=close,
            risk_adj_momentum=risk_adj_momentum,
            trend_above_ma=trend_above_ma,
            rsi=rsi,
        )

    offensive_rsi = rsi[offensive_pool] if rsi is not None else None
    defensive_rsi = rsi[defensive_pool] if rsi is not None else None
    offensive_ram = risk_adj_momentum[offensive_pool] if risk_adj_momentum is not None else None
    defensive_ram = risk_adj_momentum[defensive_pool] if risk_adj_momentum is not None else None
    offensive_trend = trend_above_ma[offensive_pool] if trend_above_ma is not None else None
    defensive_trend = trend_above_ma[defensive_pool] if trend_above_ma is not None else None

    offensive_holdings = _generate_single_engine_holdings(
        momentum_score=momentum_score[offensive_pool],
        close=close[offensive_pool],
        risk_adj_momentum=offensive_ram,
        trend_above_ma=offensive_trend,
        rsi=offensive_rsi,
    )
    defensive_holdings = _generate_single_engine_holdings(
        momentum_score=momentum_score[defensive_pool],
        close=close[defensive_pool],
        risk_adj_momentum=defensive_ram,
        trend_above_ma=defensive_trend,
        rsi=defensive_rsi,
    )

    benchmark = _cfg.DUAL_SWITCH_BENCHMARK
    ma_window = _cfg.DUAL_SWITCH_MA_WINDOW
    score_gap = _cfg.DUAL_SWITCH_SCORE_GAP
    benchmark_ma = None
    if benchmark in close.columns and ma_window > 0:
        benchmark_ma = close[benchmark].shift(1).rolling(
            ma_window,
            min_periods=max(20, ma_window // 2),
        ).mean()

    signal_dates = offensive_holdings.index.intersection(defensive_holdings.index)
    records = []
    for date in signal_dates:
        off_syms = get_holdings_on(offensive_holdings, date)
        def_syms = get_holdings_on(defensive_holdings, date)

        off_top = max((momentum_score.loc[date].get(sym, float("-inf")) for sym in off_syms), default=float("-inf"))
        def_top = max((momentum_score.loc[date].get(sym, float("-inf")) for sym in def_syms), default=float("-inf"))

        below_benchmark_ma = False
        if benchmark_ma is not None and date in benchmark_ma.index and pd.notna(benchmark_ma.loc[date]):
            below_benchmark_ma = close.loc[date, benchmark] < benchmark_ma.loc[date]

        use_defensive = below_benchmark_ma or (def_top >= off_top - score_gap)
        selected = def_syms if use_defensive else off_syms

        row = {sym: 0 for sym in all_symbols}
        for sym in selected:
            if sym in row:
                row[sym] = 1
        row["_date"] = date
        row["_n_holdings"] = len(selected)
        row["_engine"] = "defensive" if use_defensive else "offensive"
        row["_switch_reason"] = "benchmark_ma" if below_benchmark_ma else ("relative_strength" if use_defensive else "offensive_strength")
        records.append(row)

    result = pd.DataFrame(records).set_index("_date")
    result.index.name = "date"
    result = _apply_dual_engine_defensive_hold(result, close)
    if not result.empty:
        defensive_months = int(result["_engine"].astype(str).str.startswith("defensive").sum())
        logger.info(
            f"双引擎信号生成完成: {len(result)}个调仓日, "
            f"防守月份={defensive_months}, 进攻月份={len(result) - defensive_months}"
        )
    return result


def generate_monthly_holdings(
    momentum_score: pd.DataFrame,
    close: pd.DataFrame,
    risk_adj_momentum: pd.DataFrame = None,
    trend_above_ma: pd.DataFrame = None,
    rsi: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    对外统一入口。
    - single: 原始单引擎双动量
    - dual:   进攻子策略 + 防守子策略 + 月度切换层
    """
    if _cfg.QUANT_ENGINE_MODE == "dual":
        return generate_monthly_holdings_dual_engine(
            momentum_score=momentum_score,
            close=close,
            risk_adj_momentum=risk_adj_momentum,
            trend_above_ma=trend_above_ma,
            rsi=rsi,
        )
    return _generate_single_engine_holdings(
        momentum_score=momentum_score,
        close=close,
        risk_adj_momentum=risk_adj_momentum,
        trend_above_ma=trend_above_ma,
        rsi=rsi,
    )


# ─────────────────────────────────────────────
# Hybrid模式：为单个调仓日生成LLM驱动的权重
# ─────────────────────────────────────────────
def generate_llm_holdings_on_date(
    close: pd.DataFrame,
    features: dict,
    signal_date: pd.Timestamp,
    fallback_weights: dict = None,
    allowed_symbols: list = None,
    signal_engine: str = None,
    switch_reason: str = None,
    market_state: dict = None,
) -> dict:
    """
    对单个调仓信号日调用 DeepSeek 双Agent，返回 {symbol: weight}。

    参数：
    - close:            宽表收盘价
    - features:         build_features() 返回的特征字典
    - signal_date:      本次调仓信号日（收盘后产生信号）
    - fallback_weights: LLM失败时回退的量化权重
    - market_state:     HMM市场状态字典（可选）

    返回：{symbol: weight}，权重之和=1.0
    """
    try:
        from llm_agent import run_dual_agent
    except ImportError as e:
        logger.error(f"llm_agent 模块导入失败: {e}")
        return fallback_weights or {}

    return run_dual_agent(
        close=close,
        features=features,
        signal_date=signal_date,
        market_state=market_state,
        top_n=_cfg.TOP_N,
        fallback_weights=fallback_weights,
        allowed_symbols=allowed_symbols,
        signal_engine=signal_engine,
        switch_reason=switch_reason,
    )


def build_quant_anchor_weights(
    features: dict,
    quant_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    基于量化持仓表构造每个调仓日的量化锚点权重。
    这里显式复用 ATR 风险预算与静态防御分仓逻辑，
    让 hybrid 的 LLM overlay 与 quant 基线共用同一套基础仓位约束。
    """
    all_symbols = [c for c in quant_holdings.columns if not c.startswith("_")]
    atr_df = features.get("atr", pd.DataFrame())
    hist_vol_df = features.get("hist_vol", pd.DataFrame())
    close_df = features["close"]

    records = []
    for date in quant_holdings.index:
        holdings = get_holdings_on(quant_holdings, date)
        atr_row = atr_df.loc[date] if not atr_df.empty and date in atr_df.index else pd.Series(dtype=float)
        hist_vol_row = hist_vol_df.loc[date] if not hist_vol_df.empty and date in hist_vol_df.index else pd.Series(dtype=float)
        close_row = close_df.loc[date] if date in close_df.index else pd.Series(dtype=float)

        weights = compute_position_weights(
            holdings=holdings,
            atr_row=atr_row,
            close_row=close_row,
            hist_vol_row=hist_vol_row,
            circuit_breaker=None,
        )
        weights = apply_static_defensive_allocation(weights)

        row = {sym: 0.0 for sym in all_symbols}
        for sym, weight in weights.items():
            if sym in row:
                row[sym] = float(weight)
        row["_date"] = date
        records.append(row)

    anchor_df = pd.DataFrame(records).set_index("_date") if records else pd.DataFrame(columns=all_symbols)
    if not anchor_df.empty:
        anchor_df.index.name = "date"
    return anchor_df


# ─────────────────────────────────────────────
# Hybrid模式：生成带LLM权重的月度持仓表
# ─────────────────────────────────────────────
def generate_monthly_holdings_hybrid(
    close: pd.DataFrame,
    features: dict,
    quant_holdings: pd.DataFrame,
    quant_weights_df: pd.DataFrame = None,
    market_states: pd.DataFrame = None,
) -> tuple:
    """
    在纯量化持仓基础上，逐调仓日调用 DeepSeek 双Agent 覆盖权重。

    参数：
    - close:             宽表收盘价
    - features:          build_features() 返回的特征字典
    - quant_holdings:    generate_monthly_holdings() 返回的二值持仓表
    - quant_weights_df:  量化ATR权重（DataFrame或None），作为回退值
    - market_states:     compute_market_states() 返回的状态序列（可选）

    返回：(llm_holdings: pd.DataFrame 二值持仓, llm_weights: dict{date: {sym: weight}})
    """
    if quant_weights_df is None:
        quant_weights_df = build_quant_anchor_weights(features, quant_holdings)

    rebal_dates = quant_holdings.index
    all_symbols = [c for c in quant_holdings.columns if not c.startswith("_")]
    meta_cols = [c for c in quant_holdings.columns if c.startswith("_")]

    llm_weights_by_date = {}
    llm_holdings_records = []

    for date in rebal_dates:
        quant_meta = quant_holdings.loc[date, meta_cols].to_dict() if meta_cols else {}

        # 获取量化回退权重
        fallback = None
        if quant_weights_df is not None and date in quant_weights_df.index:
            fb_row = quant_weights_df.loc[date].dropna()
            fallback = fb_row[fb_row > 0].to_dict() if not fb_row.empty else None
        allowed_symbols = list(fallback.keys()) if fallback else get_holdings_on(quant_holdings, date)

        # 获取市场状态
        mkt = None
        if market_states is not None and not market_states.empty:
            avail = market_states.loc[:date]
            if not avail.empty:
                last = avail.iloc[-1]
                mkt = {
                    "state":     last.get("hmm_state", "bull"),
                    "bull_prob": last.get("hmm_bull_prob", 0.6),
                    "method":    "HMM",
                }

        logger.info(f"[Hybrid] 调用LLM决策：{date.strftime('%Y-%m-%d')}")
        weights = generate_llm_holdings_on_date(
            close=close,
            features=features,
            signal_date=date,
            fallback_weights=fallback,
            allowed_symbols=allowed_symbols,
            signal_engine=str(quant_meta.get("_engine", "single")),
            switch_reason=str(quant_meta.get("_switch_reason", "quant_overlay")),
            market_state=mkt,
        )

        llm_weights_by_date[date] = weights

        # 更新二值持仓行
        row = {sym: 0 for sym in all_symbols}
        for sym in weights:
            if sym in row:
                row[sym] = 1
        row["_date"] = date
        row["_n_holdings"] = len([sym for sym, weight in weights.items() if weight > 0])
        for meta_col in meta_cols:
            row[meta_col] = quant_meta.get(meta_col)
        row["_overlay_source"] = "llm_bounded"
        llm_holdings_records.append(row)

    if llm_holdings_records:
        llm_holdings = pd.DataFrame(llm_holdings_records).set_index("_date")
        llm_holdings.index.name = "date"
    else:
        llm_holdings = quant_holdings

    logger.info(
        f"[Hybrid] LLM持仓生成完成: {len(llm_weights_by_date)}个调仓日"
    )
    return llm_holdings, llm_weights_by_date


# ─────────────────────────────────────────────
# 便捷函数：从 holdings 二值表提取某调仓日持有ETF列表
# ─────────────────────────────────────────────
def get_holdings_on(holdings_df: pd.DataFrame, date: pd.Timestamp) -> list:
    """从月度持仓宽表中提取指定日期持有的ETF列表"""
    meta_cols = [c for c in holdings_df.columns if c.startswith("_")]
    etf_cols = [c for c in holdings_df.columns if not c.startswith("_")]

    if date not in holdings_df.index:
        # 找最近一个有效调仓日
        past = holdings_df.index[holdings_df.index <= date]
        if past.empty:
            return []
        date = past[-1]

    row = holdings_df.loc[date, etf_cols]
    return row[row == 1].index.tolist()


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_fetcher import load_data
    from feature_engine import build_features

    print("加载数据与特征...")
    close_mtx, ohlcv = load_data()
    feats = build_features(close_mtx, ohlcv)

    print("生成月度持仓信号...")
    holdings = generate_monthly_holdings(feats["momentum_score"], feats["close"])

    # 打印最近12个月持仓
    print("\n最近12个月目标持仓（1=持有）:")
    etf_cols = [c for c in holdings.columns if not c.startswith("_")]
    print(holdings[etf_cols].tail(12).to_string())

    # 统计2022年（熊市）防御资产持仓比例
    bear_market = holdings["2022-01-01":"2022-12-31"]
    defensive_months = bear_market[DEFENSIVE_ETFS].any(axis=1).sum()
    print(f"\n2022年熊市：{len(bear_market)}个月中，持有防御资产的月份 = {defensive_months}")
