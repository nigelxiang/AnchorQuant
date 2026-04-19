"""
llm_agent.py — DeepSeek 双层 Agent 协作架构

架构说明：
  Agent 1 (宏观分析师): 接收量化指标 → 判断市场环境(宏观周期/风险偏好/配置方向)
  Agent 2 (资产配置基金经理): 基于Agent1结论 → 输出 ETF持仓权重 JSON

安全要求：
  - API Key 通过 os.getenv('deepseek_api_key') 隐式加载，严禁硬编码/打印
  - 使用 python-dotenv 加载 .env 文件

使用方式：
  from llm_agent import run_dual_agent
  weights = run_dual_agent(features, close_row, signal_date, top_n=5)

LLM模式激活条件（任意满足其一）：
  - main.py --mode hybrid
  - advisor.py --advice（默认启用hybrid）
  - main.py --llm-backtest（逐月回测，慎用，API调用量大）
"""

import json
import logging
import os
import re
import sys
import hashlib
import time
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ETF_POOL, DEFENSIVE_ETFS, TOP_N, SINGLE_POS_CAP,
    LLM_CONFIG, OUTPUT_DIR, BENCHMARK_ETF,
    STYLE_PAIRS, CROSS_BORDER_REFS,
    HYBRID_MAX_REL_TILT,
)

logger = logging.getLogger(__name__)

# 加载 .env（安全：只读取 key 名，不打印值）
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)


# ─────────────────────────────────────────────
# DeepSeek API 客户端（兼容 OpenAI SDK）
# ─────────────────────────────────────────────
class DeepSeekClient:
    """
    DeepSeek API 客户端。
    使用 openai 兼容接口，base_url 指向 DeepSeek。
    API Key 通过 os.getenv 隐式加载，从不打印。
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(self):
        self._api_key = os.getenv("deepseek_api_key")
        if not self._api_key:
            raise EnvironmentError(
                "未找到 deepseek_api_key，请在 strategy/.env 文件中设置"
            )
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.DEEPSEEK_BASE_URL,
            )
        except ImportError:
            raise ImportError(
                "请安装 openai SDK：pip install openai>=1.0"
            )

    def chat(
        self,
        messages: list,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None,
    ) -> str:
        """
        调用 DeepSeek Chat API，返回回复文本。
        参数缺省时从 LLM_CONFIG 读取。
        """
        _temperature = temperature if temperature is not None else LLM_CONFIG["temperature"]
        _max_tokens  = max_tokens  if max_tokens  is not None else LLM_CONFIG["max_tokens"]
        _timeout     = timeout     if timeout     is not None else LLM_CONFIG["timeout"]

        response = self._client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            temperature=_temperature,
            max_tokens=_max_tokens,
            timeout=_timeout,
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Prompt缓存（减少重复API调用）
# ─────────────────────────────────────────────
_PROMPT_CACHE: dict = {}   # {prompt_hash: response_text}
_CACHE_LOG_PATH = os.path.join(OUTPUT_DIR, "llm_decisions.jsonl")


def _prompt_hash(messages: list) -> str:
    """对messages列表内容做MD5，用作缓存key。"""
    content = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def _log_llm_decision(
    agent_name: str,
    messages: list,
    response: str,
    signal_date: str,
    retry_count: int,
):
    """将LLM决策记录到 llm_decisions.jsonl，供后续审计分析。"""
    try:
        record = {
            "timestamp": datetime.now().isoformat(),
            "signal_date": signal_date,
            "agent": agent_name,
            "retry": retry_count,
            "response": response,
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(_CACHE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"记录LLM决策日志失败: {e}")


# ─────────────────────────────────────────────
# 数据序列化工具（量化指标 → LLM友好文本）
# ─────────────────────────────────────────────
def _build_agent1_context(
    close: pd.DataFrame,
    features: dict,
    signal_date: pd.Timestamp,
    market_state: dict = None,
) -> str:
    """
    序列化Agent1所需的宏观量化上下文。
    所有数据均使用 signal_date 当日或之前的数据，严防未来泄漏。
    """
    lines = []

    # ── 1. A股宽基状态 ──
    lines.append("【A股宽基状态】")
    benchmarks = {
        "510300": "沪深300", "159915": "创业板",
        "512100": "中证1000", "588000": "科创50",
    }
    for sym, name in benchmarks.items():
        if sym not in close.columns:
            continue
        price = close[sym].loc[:signal_date]
        if len(price) < 60:
            continue
        ret_20d = price.iloc[-1] / price.iloc[-20] - 1 if len(price) >= 20 else float("nan")
        ret_60d = price.iloc[-1] / price.iloc[-60] - 1 if len(price) >= 60 else float("nan")
        rsi_val = features.get("rsi", pd.DataFrame())
        rsi_str = ""
        if not rsi_val.empty and sym in rsi_val.columns and signal_date in rsi_val.index:
            rv = rsi_val.loc[signal_date, sym]
            if not np.isnan(rv):
                rsi_str = f", RSI={rv:.0f}"
        lines.append(
            f"  {name}({sym}): 20日={ret_20d:+.2%}, 60日={ret_60d:+.2%}{rsi_str}"
        )

    # ── 2. HMM/波动率宏观状态 ──
    lines.append("【宏观市场状态（HMM）】")
    if market_state:
        state = market_state.get("state", "unknown")
        prob  = market_state.get("bull_prob", 0.5)
        method = market_state.get("method", "")
        lines.append(f"  状态={state.upper()}, bull概率={prob:.1%} (方法:{method})")
    else:
        lines.append("  状态=UNKNOWN (HMM未启用)")

    # ── 3. A股风格比值 ──
    style_df = features.get("style_ratios", pd.DataFrame())
    if not style_df.empty and signal_date in style_df.index:
        lines.append("【A股风格比值】")
        row = style_df.loc[signal_date].dropna()
        for name, val in row.items():
            # 计算近20日变化方向
            trend_str = ""
            if signal_date in style_df.index:
                hist = style_df[name].loc[:signal_date].dropna()
                if len(hist) >= 20:
                    chg = hist.iloc[-1] / hist.iloc[-20] - 1
                    trend_str = f" (近20日变化={chg:+.2%})"
            lines.append(f"  {name}={val:.4f}{trend_str}")

    # ── 4. 跨境市场状态 ──
    cross_df = features.get("cross_border", pd.DataFrame())
    if not cross_df.empty and signal_date in cross_df.index:
        lines.append("【跨境市场近20日表现】")
        row = cross_df.loc[signal_date].dropna()
        name_map = {
            "xborder_513100": "纳指100",
            "xborder_513520": "日经225",
            "xborder_513180": "恒生科技",
            "xborder_159329": "沙特ETF",
            "xborder_159561": "德国ETF",
            "xborder_518880": "黄金ETF",
        }
        for col, val in row.items():
            pretty = name_map.get(col, col.replace("xborder_", ""))
            lines.append(f"  {pretty}: {val:+.2%}")

    # ── 5. 波动率环境 ──
    vol_df = features.get("hist_vol", pd.DataFrame())
    if not vol_df.empty and signal_date in vol_df.index:
        non_def = [c for c in vol_df.columns if c not in DEFENSIVE_ETFS]
        avg_vol = vol_df.loc[signal_date, non_def].dropna().mean()
        if not np.isnan(avg_vol):
            lines.append(f"【波动率环境】平均年化波动率={avg_vol:.1%}")

    return "\n".join(lines)


def _build_etf_ranking_table(
    features: dict,
    signal_date: pd.Timestamp,
    top_k: int = 15,
) -> str:
    """
    序列化ETF动量排名表，供Agent2选基使用。
    排除防御资产，按动量得分降序前top_k名。
    """
    mom_df = features.get("momentum_score", pd.DataFrame())
    rsi_df = features.get("rsi", pd.DataFrame())
    vol_df = features.get("hist_vol", pd.DataFrame())
    r2_df  = features.get("ols_r2", pd.DataFrame())

    if mom_df.empty or signal_date not in mom_df.index:
        return "（无动量数据）"

    score_row = mom_df.loc[signal_date]
    non_def = {s: n for s, n in ETF_POOL.items() if s not in DEFENSIVE_ETFS}
    scores = {
        s: score_row.get(s, np.nan)
        for s in non_def
        if not pd.isna(score_row.get(s, np.nan))
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    header = f"  {'排名':<4}{'代码':<8}{'名称':<14}{'动量':>8}{'RSI':>6}{'年化波动':>8}{'趋势R²':>7}"
    sep    = "  " + "-" * 56
    lines  = [header, sep]

    for rank, (sym, score) in enumerate(ranked, 1):
        name = ETF_POOL.get(sym, sym)[:7]  # 截断长名称
        rsi_val = ""
        if not rsi_df.empty and signal_date in rsi_df.index and sym in rsi_df.columns:
            rv = rsi_df.loc[signal_date, sym]
            rsi_val = f"{rv:.0f}" if not np.isnan(rv) else "--"
        vol_val = ""
        if not vol_df.empty and signal_date in vol_df.index and sym in vol_df.columns:
            vv = vol_df.loc[signal_date, sym]
            vol_val = f"{vv:.1%}" if not np.isnan(vv) else "--"
        r2_val = ""
        if not r2_df.empty and signal_date in r2_df.index and sym in r2_df.columns:
            r2v = r2_df.loc[signal_date, sym]
            r2_val = f"{r2v:.2f}" if not np.isnan(r2v) else "--"

        lines.append(
            f"  {rank:<4}{sym:<8}{name:<14}{score:>8.4f}{rsi_val:>6}{vol_val:>8}{r2_val:>7}"
        )

    # 防御资产得分
    lines.append("\n  【防御资产得分】")
    for sym in DEFENSIVE_ETFS:
        sc = score_row.get(sym, np.nan)
        name = ETF_POOL.get(sym, sym)
        sc_str = f"{sc:.4f}" if not np.isnan(sc) else "N/A"
        lines.append(f"    {sym} {name}: {sc_str}")

    return "\n".join(lines)


def _format_anchor_weights(anchor_weights: dict) -> str:
    if not anchor_weights:
        return "（无量化锚点权重）"
    lines = []
    for sym, weight in sorted(anchor_weights.items(), key=lambda item: -item[1]):
        lines.append(f"  - {sym} {ETF_POOL.get(sym, sym)}: {weight:.2%}")
    return "\n".join(lines)


def _apply_bounded_overlay(anchor_weights: dict, multipliers: dict) -> dict:
    if not anchor_weights:
        return {}

    anchor_total = sum(anchor_weights.values())
    if anchor_total <= 0:
        return {}

    adjusted = {}
    for sym, base_weight in anchor_weights.items():
        adjusted[sym] = max(0.0, base_weight * multipliers.get(sym, 1.0))

    adjusted_total = sum(adjusted.values())
    if adjusted_total <= 0:
        return dict(anchor_weights)

    return {
        sym: weight / adjusted_total * anchor_total
        for sym, weight in adjusted.items()
        if weight > 1e-8
    }


def _extract_json_object(text: str) -> dict:
    """从LLM输出中尽量稳健地提取JSON对象。"""
    cleaned = _strip_markdown_fences(text)
    candidates = [cleaned]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(cleaned[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            if isinstance(parsed.get("weights"), dict):
                return parsed["weights"]
            if isinstance(parsed.get("multipliers"), dict):
                return parsed["multipliers"]
            return parsed

    raise json.JSONDecodeError("无法从LLM输出中提取JSON对象", cleaned, 0)


def _coerce_multiplier(value) -> float | None:
    """兼容字符串、百分比等轻微格式漂移。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    if not isinstance(value, str):
        return None

    text = value.strip().lower().replace("％", "%")
    text = text.rstrip("x")
    if not text:
        return None

    if text.endswith("%"):
        try:
            return float(text[:-1].strip()) / 100.0
        except ValueError:
            return None

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def _repair_overweight_multipliers(
    anchor_weights: dict,
    multipliers: dict,
    single_cap: float,
    min_multiplier: float,
    max_multiplier: float,
    max_iter: int = 8,
) -> tuple[dict, dict, list[str]]:
    """
    若 overlay 归一化后触发单资产上限，则只压缩超限资产的乘数，
    尽量保留其余资产的 LLM 倾斜方向。

    对量化锚点本身已经超过 single_cap 的资产，不再强行压到 single_cap 以下，
    而是禁止其继续高于锚点；否则 defensive-only 月份会出现“量化锚点先天非法，LLM必败”的假异常。
    """
    anchor_total = sum(anchor_weights.values())
    repaired = dict(multipliers)
    effective_caps = {
        sym: max(single_cap, float(base_weight))
        for sym, base_weight in anchor_weights.items()
    }

    if not repaired:
        final_weights = _apply_bounded_overlay(anchor_weights, repaired)
        overweight = [
            sym for sym, weight in final_weights.items()
            if weight > effective_caps.get(sym, single_cap) + 1e-6
        ]
        return repaired, final_weights, overweight

    for _ in range(max_iter):
        final_weights = _apply_bounded_overlay(anchor_weights, repaired)
        overweight = [
            sym for sym, weight in final_weights.items()
            if weight > effective_caps.get(sym, single_cap) + 1e-6
        ]
        if not overweight:
            return repaired, final_weights, []

        changed = False
        for sym in overweight:
            base_weight = anchor_weights.get(sym, 0.0)
            cap_weight = effective_caps.get(sym, single_cap)
            current_multiplier = repaired.get(sym, 1.0)
            if base_weight <= 0:
                continue
            if cap_weight >= anchor_total:
                continue

            other_adjusted = sum(
                anchor_weights[other] * repaired.get(other, 1.0)
                for other in anchor_weights
                if other != sym
            )
            allowed_multiplier = (
                cap_weight * other_adjusted
                / ((anchor_total - cap_weight) * base_weight)
            )
            allowed_multiplier = min(max_multiplier, max(min_multiplier, allowed_multiplier))

            if allowed_multiplier + 1e-6 < current_multiplier:
                repaired[sym] = allowed_multiplier
                changed = True

        if not changed:
            break

    final_weights = _apply_bounded_overlay(anchor_weights, repaired)
    overweight = [
        sym for sym, weight in final_weights.items()
        if weight > effective_caps.get(sym, single_cap) + 1e-6
    ]
    return repaired, final_weights, overweight


# ─────────────────────────────────────────────
# Agent 1：宏观分析师
# ─────────────────────────────────────────────
def run_agent1_macro_analyst(
    client: DeepSeekClient,
    context_text: str,
    signal_date: str,
) -> dict:
    """
    Agent 1: 宏观市场环境判断
    
    输入：量化指标上下文文本
    输出：{
        "regime": str,           # 牛市/震荡/熊市/危机
        "risk_appetite": str,    # risk-on / risk-off / neutral
        "direction": str,        # 进攻/均衡/防御
        "reasoning": str,        # 一句话理由
    }
    """
    system_prompt = (
        "你是一位CFA持证的宏观策略分析师，专注于中国资本市场。"
        "请基于提供的量化数据，精准判断当前宏观周期，输出结构化JSON。"
        "严格要求：只输出纯JSON，禁止包含任何markdown标签（如```json）或额外文字。"
    )

    user_prompt = f"""请分析以下量化指标，判断当前市场宏观环境（信号日：{signal_date}）：

{context_text}

请输出如下JSON（禁止包含markdown标签，仅纯JSON）：
{{"regime":"牛市/震荡/熊市/危机之一","risk_appetite":"risk-on/neutral/risk-off之一","direction":"进攻/均衡/防御之一","reasoning":"一句话理由（30字内）"}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    retry = LLM_CONFIG["retry_count"]
    for attempt in range(retry):
        try:
            raw = client.chat(messages, temperature=0.1)
            # 清洗可能的markdown标签
            cleaned = _strip_markdown_fences(raw)
            result = json.loads(cleaned)
            # 简单校验
            if "regime" in result and "direction" in result:
                _log_llm_decision("agent1_macro", messages, raw, signal_date, attempt)
                logger.info(
                    f"[Agent1] 宏观状态: {result.get('regime')} | "
                    f"方向: {result.get('direction')} | "
                    f"理由: {result.get('reasoning', '')}"
                )
                return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Agent1 解析失败 (attempt {attempt+1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(1)

    # 全部失败：返回保守默认值
    logger.error("Agent1 全部尝试失败，返回保守默认: 均衡/防御")
    return {
        "regime": "未知",
        "risk_appetite": "neutral",
        "direction": "均衡",
        "reasoning": "LLM调用失败，保守默认",
    }


# ─────────────────────────────────────────────
# Agent 2：资产配置基金经理
# ─────────────────────────────────────────────
def run_agent2_portfolio_manager(
    client: DeepSeekClient,
    agent1_result: dict,
    etf_table: str,
    signal_date: str,
    anchor_weights: dict = None,
    allowed_symbols: list = None,
    signal_engine: str = None,
    switch_reason: str = None,
    top_n: int = TOP_N,
    single_cap: float = SINGLE_POS_CAP,
    max_rel_tilt: float = HYBRID_MAX_REL_TILT,
) -> dict:
    """
    Agent 2: 基于宏观结论选ETF并分配权重
    
    输出：{symbol: weight}，权重之和=1.0，合法ETF代码
    """
    direction = agent1_result.get("direction", "均衡")
    regime    = agent1_result.get("regime", "未知")
    reasoning = agent1_result.get("reasoning", "")
    base_weights = dict(anchor_weights or {})
    if not base_weights:
        allowed = list(allowed_symbols or [])
        if not allowed:
            allowed = list(DEFENSIVE_ETFS)
        eq_weight = 1.0 / len(allowed)
        base_weights = {sym: eq_weight for sym in allowed}
    allowed = list(base_weights.keys())
    min_multiplier = max(0.0, 1.0 - max_rel_tilt)
    max_multiplier = 1.0 + max_rel_tilt

    # 防御指令：若宏观判断为"危机/熊市"且建议"防御"
    is_defensive = (
        direction == "防御"
        or regime in ("危机", "熊市")
    )

    defense_instruction = (
        "⚠️ 宏观分析师判断当前市场为防御模式，"
        "请将至少60%仓位配置在防御资产（511260、511880、518880），"
        "其余仓位选动量已正转的ETF。"
        if is_defensive else
        "宏观偏进攻/均衡，重点选配动量强劲且R²高（趋势稳定）的ETF。"
    )

    system_prompt = (
        "你是一位管理百亿ETF轮动组合的资产配置基金经理，专注于中国A股和QDII市场。"
        "你的任务不是重建组合，而是在量化基线组合上做受限倾斜。"
        "严格要求：只能在给定ETF集合内调整，禁止新增ETF。"
        "严格要求：只输出纯JSON乘数，禁止包含任何markdown标签（如```json）或说明文字。"
    )

    user_prompt = f"""信号日期：{signal_date}

【宏观分析师结论】
- 市场状态：{regime}
- 风险偏好：{agent1_result.get('risk_appetite', 'neutral')}
- 配置方向：{direction}
- 分析理由：{reasoning}

【量化基线】
- 当前月度引擎：{signal_engine or 'single'}
- 月度切换原因：{switch_reason or 'quant_base'}
- 允许调整的ETF集合（禁止新增）: {', '.join(f'{sym}({ETF_POOL.get(sym, sym)})' for sym in allowed)}
- 量化基线权重：
{_format_anchor_weights(base_weights)}

{defense_instruction}

【可选ETF动量排行（前15名，供参考）】
{etf_table}

【约束条件】
- 只能对量化基线中的ETF做倾斜，不能新增ETF
- 对每个ETF输出一个乘数，范围必须在 [{min_multiplier:.2f}, {max_multiplier:.2f}] 内
- 最终归一化后的单资产权重不能超过 {single_cap:.0%}；若某资产基线权重已接近上限，优先保持 1.0 或下调
- 若观点不强，对该ETF输出 1.0
- 系统会用“量化基线权重 × 乘数”后自动归一化为最终权重

请输出乘数JSON（仅纯JSON，无markdown标签，无说明文字）：
{{"ETF代码1": 1.00, "ETF代码2": 0.95, ...}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    retry = LLM_CONFIG["retry_count"]
    for attempt in range(retry):
        try:
            raw = client.chat(messages, temperature=0.1)
            multipliers = _extract_json_object(raw)

            valid_multipliers = {}
            for sym, multiplier in multipliers.items():
                sym = str(sym).strip()
                multiplier_value = _coerce_multiplier(multiplier)
                if sym not in base_weights or multiplier_value is None:
                    continue
                valid_multipliers[sym] = min(max(multiplier_value, min_multiplier), max_multiplier)

            if not valid_multipliers:
                raise ValueError("没有合法overlay乘数")

            for sym in base_weights:
                valid_multipliers.setdefault(sym, 1.0)

            repaired_multipliers, final_weights, overweight = _repair_overweight_multipliers(
                anchor_weights=base_weights,
                multipliers=valid_multipliers,
                single_cap=single_cap,
                min_multiplier=min_multiplier,
                max_multiplier=max_multiplier,
            )
            if not final_weights:
                raise ValueError("overlay后权重为空")

            if overweight:
                raise ValueError("overlay后单资产权重越界")

            if repaired_multipliers != valid_multipliers:
                logger.info(
                    f"[Agent2] 已按单资产上限回修乘数: 原始={valid_multipliers} | 回修后={repaired_multipliers}"
                )
            valid_multipliers = repaired_multipliers

            _log_llm_decision("agent2_portfolio", messages, raw, signal_date, attempt)
            logger.info(
                f"[Agent2] Overlay乘数: {valid_multipliers} | 最终权重: {final_weights}"
            )
            return final_weights

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Agent2 解析失败 (attempt {attempt+1}/{retry}): {e}")
            if attempt < retry - 1:
                time.sleep(1)

    # 全部失败：返回 None，由调用方回退到量化信号
    logger.error("Agent2 全部尝试失败，将回退到纯量化信号")
    return None


# ─────────────────────────────────────────────
# 公开接口：运行双Agent决策
# ─────────────────────────────────────────────
def run_dual_agent(
    close: pd.DataFrame,
    features: dict,
    signal_date: pd.Timestamp,
    market_state: dict = None,
    top_n: int = TOP_N,
    fallback_weights: dict = None,
    allowed_symbols: list = None,
    signal_engine: str = None,
    switch_reason: str = None,
) -> dict:
    """
    运行 Agent1 + Agent2 协作决策，返回 ETF权重字典。

    参数：
    - close:           宽表收盘价 DataFrame
    - features:        build_features() 返回的特征字典
    - signal_date:     本次调仓信号日期
    - market_state:    get_latest_market_state() 返回的字典（可选）
    - top_n:           目标持仓数量
    - fallback_weights: LLM失败时的回退权重（通常是纯量化ATR权重）

    返回：{symbol: weight}，权重之和=1.0
    失败时返回 fallback_weights（若有）或等权防御仓位
    """
    date_str = signal_date.strftime("%Y-%m-%d") if hasattr(signal_date, "strftime") else str(signal_date)

    # ── Prompt缓存检查 ──
    cache_key = _prompt_hash([
        date_str,
        LLM_CONFIG["model"],
        sorted(list((fallback_weights or {}).keys())),
        fallback_weights or {},
        sorted(list(allowed_symbols or [])),
        signal_engine or "single",
        switch_reason or "quant_base",
    ])
    if LLM_CONFIG.get("cache_enabled") and cache_key in _PROMPT_CACHE:
        logger.info(f"[LLM缓存命中] signal_date={date_str}")
        return _PROMPT_CACHE[cache_key]

    # ── 初始化客户端 ──
    try:
        client = DeepSeekClient()
    except (EnvironmentError, ImportError) as e:
        logger.error(f"DeepSeek客户端初始化失败: {e}")
        return fallback_weights or _default_defense_weights()

    # ── Agent 1：宏观分析师 ──
    context_text = _build_agent1_context(close, features, signal_date, market_state)
    agent1_result = run_agent1_macro_analyst(client, context_text, date_str)

    # ── Agent 2：资产配置基金经理 ──
    etf_table = _build_etf_ranking_table(features, signal_date, top_k=15)
    agent2_weights = run_agent2_portfolio_manager(
        client=client,
        agent1_result=agent1_result,
        etf_table=etf_table,
        signal_date=date_str,
        anchor_weights=fallback_weights,
        allowed_symbols=allowed_symbols,
        signal_engine=signal_engine,
        switch_reason=switch_reason,
        top_n=top_n,
    )

    if agent2_weights is None:
        logger.warning("双Agent决策失败，回退到量化信号")
        result = fallback_weights or _default_defense_weights()
    else:
        result = agent2_weights
        if LLM_CONFIG.get("cache_enabled"):
            _PROMPT_CACHE[cache_key] = result

    return result


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def _strip_markdown_fences(text: str) -> str:
    """清洗LLM输出中可能包含的markdown代码块标签。"""
    # 移除 ```json ... ``` 或 ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    # 移除前后空白
    return text.strip()


def _default_defense_weights() -> dict:
    """返回等权防御仓位（LLM不可用时的最后兜底）。"""
    n = len(DEFENSIVE_ETFS)
    return {sym: round(1.0 / n, 4) for sym in DEFENSIVE_ETFS}


# ─────────────────────────────────────────────
# 独立测试入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=== DeepSeek 双Agent 测试 ===")

    from data_fetcher import load_data
    from feature_engine import build_features
    from market_state import get_latest_market_state

    print("加载数据和特征...")
    close_mtx, ohlcv = load_data()
    features = build_features(close_mtx, ohlcv)

    signal_date = features["close"].index[-1]
    print(f"信号日期: {signal_date.strftime('%Y-%m-%d')}")

    print("获取市场状态...")
    mkt_state = get_latest_market_state(close_mtx)
    print(f"  市场状态: {mkt_state}")

    print("运行双Agent决策...")
    weights = run_dual_agent(
        close=close_mtx,
        features=features,
        signal_date=signal_date,
        market_state=mkt_state,
        top_n=TOP_N,
    )

    print("\n最终持仓权重:")
    for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
        name = ETF_POOL.get(sym, sym)
        print(f"  {sym} {name:<12}: {w:.2%}")
    print(f"权重之和: {sum(weights.values()):.4f}")
