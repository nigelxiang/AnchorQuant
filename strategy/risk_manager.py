"""
risk_manager.py — 仓位管理与风控
包含三层风控：
  1. ATR 等风险预算（Risk Parity）：每只ETF风险贡献相等
  2. 目标波动率缩放（Target Volatility）：组合年化波动率控制在 TARGET_VOL
  3. 最大回撤熔断（Circuit Breaker）：回撤超 MAX_DD_TRIGGER 时全转防御
"""

import numpy as np
import pandas as pd
import logging

from config import (
    ATR_WINDOW, TARGET_VOL, MAX_DD_TRIGGER,
    SINGLE_POS_CAP, TOTAL_INVEST_RATIO,
    DEFENSIVE_ETFS, TRADING_DAYS_PER_YEAR,
    CB_TIMEOUT_MONTHS, CB_TIER1_DD, CB_TIER2_DD,
    RISK_RATIO,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. ATR 等风险预算仓位计算
# ─────────────────────────────────────────────
def calc_atr_weights(
    holdings: list,
    atr_row: pd.Series,
    close_row: pd.Series,
) -> dict:
    """
    根据 ATR 做等风险预算分配仓位。
    
    原理：weight_i = (1/ATR_i) / sum(1/ATR_j)
    即波动率越高 → ATR越大 → 仓位越小，实现各ETF风险贡献均等。
    
    入参：
    - holdings: 本期持有ETF列表
    - atr_row:  当日各ETF的ATR值（pd.Series, index=symbol）
    - close_row: 当日各ETF收盘价（pd.Series, index=symbol）
    
    返回：{symbol: weight}，权重之和 <= TOTAL_INVEST_RATIO
    """
    if not holdings:
        return {}

    inv_atr = {}
    for sym in holdings:
        atr_val = atr_row.get(sym, np.nan)
        close_val = close_row.get(sym, np.nan)
        if pd.isna(atr_val) or atr_val <= 0 or pd.isna(close_val) or close_val <= 0:
            # ATR缺失时退化为等权
            inv_atr[sym] = 1.0
        else:
            # 归一化ATR（ATR / Close = 相对波动率，更具可比性）
            rel_atr = atr_val / close_val
            inv_atr[sym] = 1.0 / rel_atr if rel_atr > 0 else 1.0

    total_inv = sum(inv_atr.values())
    if total_inv == 0:
        # 全部退化为等权
        n = len(holdings)
        return {sym: TOTAL_INVEST_RATIO / n for sym in holdings}

    raw_weights = {sym: v / total_inv for sym, v in inv_atr.items()}

    # 单只ETF仓位上限
    capped = {}
    overflow = 0.0
    uncapped = []
    for sym, w in raw_weights.items():
        if w > SINGLE_POS_CAP:
            capped[sym] = SINGLE_POS_CAP
            overflow += w - SINGLE_POS_CAP
        else:
            uncapped.append(sym)
            capped[sym] = w

    # 将溢出部分按比例分配给未封顶的ETF
    if overflow > 0 and uncapped:
        uncapped_total = sum(capped[s] for s in uncapped)
        for sym in uncapped:
            capped[sym] += overflow * (capped[sym] / uncapped_total)

    # 整体缩放 TOTAL_INVEST_RATIO
    weight_sum = sum(capped.values())
    final_weights = {sym: w / weight_sum * TOTAL_INVEST_RATIO for sym, w in capped.items()}

    return final_weights


# ─────────────────────────────────────────────
# 2. 目标波动率缩放
# ─────────────────────────────────────────────
def scale_to_target_vol(
    weights: dict,
    hist_vol: pd.Series,
    target_vol: float = None,
) -> dict:
    """
    若当前组合预估波动率 > target_vol，则等比例缩减所有仓位。
    组合波动率用各ETF历史波动率的加权平均近似（忽略相关性，保守估计）。
    
    返回缩放后的仓位字典。
    """
    if target_vol is None:
        target_vol = TARGET_VOL
    if not weights:
        return {}

    # 计算组合预估波动率（加权平均波动率，无相关性修正 → 偏高估，保守）
    portfolio_vol = sum(
        w * hist_vol.get(sym, target_vol)
        for sym, w in weights.items()
    ) / sum(weights.values()) if weights else target_vol

    if portfolio_vol <= 0:
        return weights

    if portfolio_vol > target_vol:
        scale = target_vol / portfolio_vol
        scaled = {sym: w * scale for sym, w in weights.items()}
        logger.debug(f"目标波动率缩放: 预估组合波动率={portfolio_vol:.1%}, 缩放比例={scale:.2f}")
        return scaled

    return weights


def apply_static_defensive_allocation(weights: dict) -> dict:
    """
    将常驻防御仓预算注入目标权重。
    该逻辑原先只存在于回测执行层，这里抽成公共函数，
    以便量化锚点、LLM回退权重和实际执行使用完全一致的静态防御分仓规则。
    """
    if not weights:
        return {}

    if RISK_RATIO >= 1.0:
        return dict(weights)

    scaled = {
        sym: w * RISK_RATIO
        for sym, w in weights.items()
        if sym not in DEFENSIVE_ETFS
    }
    def_budget = TOTAL_INVEST_RATIO * (1 - RISK_RATIO)
    for sym in DEFENSIVE_ETFS:
        existing = weights.get(sym, 0.0) * RISK_RATIO
        scaled[sym] = existing + def_budget / len(DEFENSIVE_ETFS)

    return scaled


# ─────────────────────────────────────────────
# 3. 回撤风控：分档熔断（替代二元开关）
# ─────────────────────────────────────────────
class CircuitBreaker:
    """
    分档组合回撤监控：
    - 回撤 < CB_TIER1_DD (8%)  : 全风险（risk_ratio = 1.0）
    - CB_TIER1_DD ≤ 回撤 < CB_TIER2_DD (12%)  : risk_ratio = 0.70（30%防御）
    - CB_TIER2_DD ≤ 回撤 < MAX_DD_TRIGGER (16%): risk_ratio = 0.40（60%防御）
    - 回撤 ≥ MAX_DD_TRIGGER (16%): risk_ratio = 0.0（全防御，硬熔断）

    分档设计优势：
    - 减少二元开关引起的"来回切换"损耗（whipsaw）
    - 小回撤时轻度对冲，大回撤时完全防御
    - 独立超时机制：仅硬熔断（0%仓位）适用超时自动重置
    """

    TIERS = [
        # (drawdown_threshold, risk_ratio) — 按严重程度从大到小排列
        (MAX_DD_TRIGGER,  0.00),   # ≥16%：全防御（硬熔断）
        (CB_TIER2_DD,     0.40),   # ≥12%：60%防御
        (CB_TIER1_DD,     0.70),   # ≥8%：30%防御
        (0.00,            1.00),   # < 8%：正常
    ]

    def __init__(self, trigger: float = None):
        self.trigger = trigger if trigger is not None else MAX_DD_TRIGGER
        self.peak_value = None
        self.risk_ratio = 1.0          # 当前风险资产比例 (0.0 ~ 1.0)
        self.is_triggered = False      # 向后兼容：True 仅代表硬熔断（0%风险）
        self.trigger_date = None       # 硬熔断触发时间（用于超时重置）

    def _calc_risk_ratio(self, drawdown: float) -> float:
        """根据当前回撤深度确定风险比例。"""
        dd_abs = abs(drawdown)
        for threshold, ratio in self.TIERS:
            if dd_abs >= threshold:
                return ratio
        return 1.0

    def update(self, current_nav: float, current_date=None) -> bool:
        if self.peak_value is None:
            self.peak_value = current_nav

        # 硬熔断超时重置（仅在全防御时适用）
        if self.is_triggered and self.trigger_date is not None and current_date is not None:
            months = (
                (current_date.year - self.trigger_date.year) * 12
                + (current_date.month - self.trigger_date.month)
            )
            if months >= CB_TIMEOUT_MONTHS:
                self.is_triggered = False
                self.trigger_date = None
                self.peak_value = current_nav
                self.risk_ratio = 1.0
                logger.info(f"熔断超时重置：已持续{months}个月，净值={current_nav:.0f}")

        # 更新全时峰值（仅在非全防御时跟踪）
        if not self.is_triggered and current_nav > self.peak_value:
            self.peak_value = current_nav

        # 计算当前回撤
        dd = (current_nav - self.peak_value) / self.peak_value

        # 确定风险档位
        new_ratio = self._calc_risk_ratio(dd)

        # 硬熔断恢复条件：回撤缩小到 TIER2 以下才能退出全防御
        if self.is_triggered:
            if abs(dd) < CB_TIER2_DD:
                self.is_triggered = False
                self.trigger_date = None
                logger.info(f"硬熔断恢复：回撤={dd:.1%}，净值={current_nav:.0f}")
            # 在硬熔断中暂时不更新 risk_ratio（保持 0）
        else:
            # 进入硬熔断
            if new_ratio == 0.0:
                self.is_triggered = True
                self.trigger_date = current_date
                logger.warning(f"硬熔断触发！回撤={dd:.1%}，净值={current_nav:.0f}，峰值={self.peak_value:.0f}")
            self.risk_ratio = new_ratio

        return self.is_triggered

    def get_current_risk_ratio(self) -> float:
        """返回当前风险资产比例（0.0 ~ 1.0）。"""
        if self.is_triggered:
            return 0.0
        return self.risk_ratio

    def get_defensive_weights(self, defensive_etfs=None):
        if defensive_etfs is None:
            defensive_etfs = DEFENSIVE_ETFS
        n = len(defensive_etfs)
        w = TOTAL_INVEST_RATIO / n
        return {sym: w for sym in defensive_etfs}

    def blend_weights(self, risky_weights: dict, defensive_etfs=None):
        """
        按当前风险比例混合风险仓位与防御仓位。
        - risk_ratio = 1.0 → 完全风险仓位（不变）
        - risk_ratio = 0.0 → 完全防御仓位
        - 中间值 → 按比例混合
        """
        rr = self.get_current_risk_ratio()
        if rr >= 1.0:
            return risky_weights
        if rr <= 0.0:
            return self.get_defensive_weights(defensive_etfs)

        # 部分混合
        def_weights = self.get_defensive_weights(defensive_etfs)
        blended = {}
        # 风险仓位缩放
        for sym, w in risky_weights.items():
            blended[sym] = w * rr
        # 防御仓位补充
        def_share = 1.0 - rr
        for sym, w in def_weights.items():
            blended[sym] = blended.get(sym, 0.0) + w * def_share
        return blended


# ─────────────────────────────────────────────
# 统一仓位计算入口
# ─────────────────────────────────────────────
def compute_position_weights(
    holdings: list,
    atr_row: pd.Series,
    close_row: pd.Series,
    hist_vol_row: pd.Series,
    circuit_breaker: CircuitBreaker = None,
    current_nav: float = None,
) -> dict:
    """
    完整仓位计算流程：
    1. 检查熔断状态
    2. ATR 等风险预算
    3. 目标波动率缩放

    返回 {symbol: weight}
    """
    # 1. 熔断检查
    if circuit_breaker is not None and current_nav is not None:
        if circuit_breaker.update(current_nav):
            return circuit_breaker.get_defensive_weights()

    # 2. ATR 等风险预算
    weights = calc_atr_weights(holdings, atr_row, close_row)
    if not weights:
        return circuit_breaker.get_defensive_weights() if circuit_breaker else {}

    # 3. 目标波动率缩放
    weights = scale_to_target_vol(weights, hist_vol_row)

    return weights
