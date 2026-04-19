"""
advisor.py — 投资建议生成与邮件发送

功能：
1. 基于最新数据，生成当日目标持仓和仓位建议
2. 在调仓前一个交易日，生成含挂单价格的操作建议
3. 输出详细的投资建议报告（含动量得分、风控状态、换仓建议）
4. 可选：通过 SMTP 发送邮件（配置读取自 .env 文件）

使用方法：
    python advisor.py                             # 直接生成建议（终端输出）
    python advisor.py --mode hybrid               # 输出量化建议 + LLM建议（并列展示）
    python main.py --advice                       # 在回测后追加生成建议
    python main.py --mode hybrid --advice         # hybrid模式下保留量化建议，并附带LLM建议

邮箱参数（写入 strategy/.env 文件）：
    SMTP_HOST     : SMTP服务器，如 smtp.163.com
    SMTP_PORT     : SMTP端口，如 465（SSL）或 587（STARTTLS）
    SMTP_USER     : 发件邮箱
    SMTP_PASS     : 邮箱授权码（非登录密码）
    EMAIL_FROM    : 发件人地址（默认=SMTP_USER）
    EMAIL_TO      : 收件人地址（逗号分隔多个）
"""

import os
import sys
import logging
import smtplib
import textwrap
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

import numpy as np
import pandas as pd

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ETF_POOL, DEFENSIVE_ETFS, TOP_N, OUTPUT_DIR,
    CB_TIER1_DD, CB_TIER2_DD, MAX_DD_TRIGGER,
    MOM_WINDOWS, MOM_WEIGHTS, SKIP_DAYS,
)
from signal_generator import generate_monthly_holdings, get_holdings_on

logger = logging.getLogger(__name__)

# 加载 .env 文件
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)


def load_email_config() -> dict:
    """从 .env 文件读取邮箱配置，返回字典。"""
    return {
        "smtp_host": os.getenv("SMTP_HOST", ""),
        "smtp_port": int(os.getenv("SMTP_PORT", "465")),
        "smtp_user": os.getenv("SMTP_USER", ""),
        "smtp_pass": os.getenv("SMTP_PASS", ""),
        "email_from": os.getenv("EMAIL_FROM", "") or os.getenv("SMTP_USER", ""),
        "email_to": os.getenv("EMAIL_TO", ""),
    }


def _classify_smtp_auth_error(exc: Exception) -> str:
    """根据 SMTP 返回内容细分认证失败原因。"""
    message = str(exc).upper()

    if "PASSERR" in message or "PASSWORD" in message or "AUTHENTICATION FAILED" in message:
        return "授权码或密码错误"

    if (
        "CLIENT" in message
        or "POLICY" in message
        or "DISABLED" in message
        or "NOT PERMITTED" in message
        or "NOT ALLOWED" in message
        or "APPLICATION-SPECIFIC PASSWORD REQUIRED" in message
        or "5.7.1" in message
        or "5.7.14" in message
    ):
        return "企业邮箱策略拦截或未开通 SMTP/客户端登录"

    return "SMTP 认证失败，可能是授权码错误或企业邮箱策略限制"


def _classify_smtp_disconnect_error(exc: Exception) -> str:
    """细分服务器主动断开连接的场景。"""
    message = str(exc).upper()

    if "UNEXPECTEDLY CLOSED" in message or "DISCONNECTED" in message:
        return "SMTP 服务器在认证阶段主动断开连接，通常是企业邮箱策略拦截，或错误授权码触发了服务端风控"

    return f"SMTP 连接被服务器断开：{exc}"


def test_smtp_login(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
) -> bool:
    """仅测试 SMTP 连接与登录，不发送邮件。"""
    if not all([smtp_host, smtp_port, smtp_user, smtp_pass]):
        logger.error("SMTP 自检失败：配置不完整，请检查 .env 或命令行参数")
        return False

    try:
        if smtp_port == 465:
            logger.info(f"SMTP 自检：使用 SSL 连接 {smtp_host}:{smtp_port}")
            import ssl
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=30) as server:
                server.ehlo()
                server.login(smtp_user, smtp_pass)
        elif smtp_port == 587:
            logger.info(f"SMTP 自检：使用 STARTTLS 连接 {smtp_host}:{smtp_port}")
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_user, smtp_pass)
        else:
            logger.info(f"SMTP 自检：使用普通 SMTP 连接 {smtp_host}:{smtp_port}")
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                server.login(smtp_user, smtp_pass)

        logger.info("SMTP 自检成功：连接与登录均正常")
        return True
    except smtplib.SMTPAuthenticationError as e:
        reason = _classify_smtp_auth_error(e)
        logger.error(f"SMTP 自检失败：{reason}")
        logger.error(f"SMTP 返回：{e}")
    except smtplib.SMTPServerDisconnected as e:
        logger.error(f"SMTP 自检失败：{_classify_smtp_disconnect_error(e)}")
    except smtplib.SMTPConnectError:
        logger.error(f"SMTP 自检失败：无法连接 {smtp_host}:{smtp_port}")
    except Exception as e:
        logger.error(f"SMTP 自检失败：{e}")

    return False


def is_latest_trading_day_today(
    trading_dates: pd.DatetimeIndex,
    now: datetime = None,
) -> bool:
    """判断最新交易日是否就是今天，用于 cron 安全执行。"""
    if trading_dates is None or len(trading_dates) == 0:
        return False
    if now is None:
        now = datetime.today()
    return trading_dates[-1].normalize() == pd.Timestamp(now.date())


# ─────────────────────────────────────────────
# 判断今日是否为调仓前一个交易日
# ─────────────────────────────────────────────
def get_next_rebalance_exec_date(trading_dates: pd.DatetimeIndex, rebal_signal_dates: pd.DatetimeIndex) -> dict:
    """
    建立映射：对每个信号日（调仓信号产生日），找到执行日（T+1）。
    返回 {signal_date: execution_date}
    """
    td_list = trading_dates.sort_values().tolist()
    mapping = {}
    for sd in rebal_signal_dates:
        future = [d for d in td_list if d > sd]
        if future:
            mapping[sd] = future[0]
    return mapping


def is_pre_rebalance_day(
    today: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    rebal_signal_dates: pd.DatetimeIndex,
) -> tuple:
    """
    判断 today 是否是某个调仓的信号日（即调仓前一个交易日的收盘后）。
    信号日收盘后产生调仓信号，次交易日（执行日）按挂单价执行。

    返回：(is_signal_day: bool, execution_date: pd.Timestamp or None)
    """
    if today in rebal_signal_dates:
        td_list = trading_dates.sort_values().tolist()
        future = [d for d in td_list if d > today]
        if future:
            return True, future[0]
    return False, None


# ─────────────────────────────────────────────
# 挂单价格计算
# ─────────────────────────────────────────────
def compute_limit_prices(
    target_weights: dict,
    current_weights: dict,
    close_row: pd.Series,
    buy_premium: float = 0.002,
    sell_discount: float = 0.002,
) -> list:
    """
    根据目标权重和当前持仓，计算每只ETF的挂单价（限价单）。

    逻辑：
    - 买入（新增或加仓）：挂单价 = 收盘价 × (1 + buy_premium)，略高于收盘价以提高成交概率
    - 卖出（清仓或减仓）：挂单价 = 收盘价 × (1 - sell_discount)，略低于收盘价以确保成交
    - 持仓不变：不生成挂单

    返回：[{symbol, name, action, close_price, limit_price, target_weight, change_pct}, ...]
    """
    orders = []

    # 所有相关 symbol（目标 + 当前持仓中的）
    all_syms = set(list(target_weights.keys()) + list(current_weights.keys()))

    for sym in sorted(all_syms):
        tw = target_weights.get(sym, 0.0)
        cw = current_weights.get(sym, 0.0)
        close_price = close_row.get(sym, np.nan)
        name = ETF_POOL.get(sym, sym)

        if pd.isna(close_price) or close_price <= 0:
            continue

        delta = tw - cw
        # 仅当权重变化超过 1% 时才生成挂单
        if abs(delta) < 0.01:
            continue

        if delta > 0:
            action = "买入"
            limit_price = round(close_price * (1 + buy_premium), 3)
        else:
            action = "卖出"
            limit_price = round(close_price * (1 - sell_discount), 3)

        orders.append({
            "symbol": sym,
            "name": name,
            "action": action,
            "close_price": close_price,
            "limit_price": limit_price,
            "target_weight": tw,
            "current_weight": cw,
            "change_pct": delta,
        })

    return orders


# ─────────────────────────────────────────────
# 核心：基于最新数据计算当日建议
# ─────────────────────────────────────────────
def generate_advice(
    close_matrix: pd.DataFrame,
    features: dict,
    nav_df: pd.DataFrame = None,
    trade_df: pd.DataFrame = None,
    monthly_holdings: pd.DataFrame = None,
    current_capital: float = 1_000_000.0,
    use_llm: bool = False,
) -> dict:
    """
    生成当日投资建议字典。

    参数：
    - use_llm: True时调用DeepSeek双Agent生成LLM权重建议（hybrid模式）

    返回结构：
    {
        "date": str,           # 建议日期
        "market_regime": str,  # 市场状态简述
        "holdings": list,      # 建议持仓列表
        "weights": dict,       # 量化建议权重 {symbol: weight}
        "scores": dict,        # 各ETF动量得分
        "circuit_status": str, # 熔断状态
        "nav_latest": float,   # 最新净值
        "drawdown_pct": float, # 当前回撤
        "action_items": list,  # 具体操作建议
        "report_text": str,    # 完整文字报告
        "llm_weights": dict,   # LLM建议权重（hybrid模式，否则为None）
    }
    """
    today = datetime.today().strftime("%Y-%m-%d")
    close = features["close"]
    momentum_score = features["momentum_score"]
    hist_vol = features["hist_vol"]
    atr_df = features.get("atr", pd.DataFrame())

    # ── 最新可用信号日（最后一个有数据的调仓日期）──
    last_signal_date = momentum_score.dropna(how="all").index[-1]
    score_row = momentum_score.loc[last_signal_date]
    close_row = close.loc[last_signal_date] if last_signal_date in close.index else close.iloc[-1]

    # ── 全 ETF 得分表（排除防御资产）──
    all_scores = {}
    for sym in ETF_POOL:
        if sym in score_row.index and not pd.isna(score_row[sym]):
            all_scores[sym] = score_row[sym]

    # 按得分排序
    sorted_etfs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    # ── 防御资产得分（绝对动量基准线）──
    def_scores = {s: score_row.get(s, np.nan) for s in DEFENSIVE_ETFS if s in score_row.index}
    def_threshold = max((v for v in def_scores.values() if not pd.isna(v)), default=0.0)

    # ── 净值与回撤状态 ──
    nav_latest = current_capital
    drawdown_pct = 0.0
    if nav_df is not None and not nav_df.empty:
        nav_latest = nav_df["nav"].iloc[-1]
        peak = nav_df["nav"].cummax().iloc[-1]
        drawdown_pct = (nav_latest - peak) / peak if peak > 0 else 0.0
        is_circuit_active = bool(nav_df["is_circuit"].iloc[-1])
    else:
        is_circuit_active = False

    # ── 确定熔断档位 ──
    dd_abs = abs(drawdown_pct)
    if dd_abs >= MAX_DD_TRIGGER or is_circuit_active:
        circuit_status = f"🔴 硬熔断（回撤={drawdown_pct:.1%}，≥{MAX_DD_TRIGGER:.0%}）— 全防御模式"
        risk_ratio = 0.0
    elif dd_abs >= CB_TIER2_DD:
        circuit_status = f"🟠 第二档警戒（回撤={drawdown_pct:.1%}，≥{CB_TIER2_DD:.0%}）— 40%风险仓"
        risk_ratio = 0.40
    elif dd_abs >= CB_TIER1_DD:
        circuit_status = f"🟡 第一档警戒（回撤={drawdown_pct:.1%}，≥{CB_TIER1_DD:.0%}）— 70%风险仓"
        risk_ratio = 0.70
    else:
        circuit_status = f"🟢 正常（回撤={drawdown_pct:.1%}，< {CB_TIER1_DD:.0%}）— 全仓运行"
        risk_ratio = 1.0

    # ── 建议持仓（与上月调仓日保持一致，或实时计算）──
    if monthly_holdings is not None and not monthly_holdings.empty:
        # 使用最近一次调仓信号的持仓
        signal_cols = [c for c in monthly_holdings.columns if not c.startswith("_")]
        latest_holding_row = monthly_holdings[signal_cols].iloc[-1]
        suggested_holdings = latest_holding_row[latest_holding_row == 1].index.tolist()
        signal_source = f"最近调仓日（{monthly_holdings.index[-1].strftime('%Y-%m-%d')}）"
    else:
        # 用完整信号生成流水线实时计算（与回测逻辑完全一致：RSI过滤 + 相关性过滤 + 绝对动量）
        live_holdings = generate_monthly_holdings(
            momentum_score=momentum_score,
            close=close,
            rsi=features.get("rsi"),
        )
        if not live_holdings.empty:
            signal_cols = [c for c in live_holdings.columns if not c.startswith("_")]
            latest_holding_row = live_holdings[signal_cols].iloc[-1]
            suggested_holdings = latest_holding_row[latest_holding_row == 1].index.tolist()
            signal_source = f"实时信号（最新调仓日 {live_holdings.index[-1].strftime('%Y-%m-%d')}）"
        else:
            suggested_holdings = [DEFENSIVE_ETFS[0]]
            signal_source = "实时计算（数据不足，持防御资产）"

    # ── 根据熔断档位调整最终持仓（熔断时替换为防御资产）──
    if risk_ratio == 0.0:
        final_holdings = DEFENSIVE_ETFS[:2]
        action_note = "熔断期间持有防御资产，待回撤收窄至12%以下后再考虑恢复"
    else:
        final_holdings = suggested_holdings

    # ── 建议权重（简化 ATR 等权） ──
    n = len(final_holdings)
    if n > 0 and atr_df is not None and not atr_df.empty and last_signal_date in atr_df.index:
        atr_row = atr_df.loc[last_signal_date]
        inv_atr = {}
        for sym in final_holdings:
            a = atr_row.get(sym, np.nan)
            p = close_row.get(sym, np.nan)
            if not pd.isna(a) and a > 0 and not pd.isna(p) and p > 0:
                inv_atr[sym] = 1.0 / (a / p)
            else:
                inv_atr[sym] = 1.0
        total = sum(inv_atr.values())
        raw_weights = {sym: v / total for sym, v in inv_atr.items()}
        # 应用熔断缩放
        weights = {sym: w * risk_ratio for sym, w in raw_weights.items()}
        # 防御补充
        if risk_ratio < 1.0:
            def_budget = (1.0 - risk_ratio) * 0.99
            best_def = DEFENSIVE_ETFS[0]
            weights[best_def] = weights.get(best_def, 0.0) + def_budget
    else:
        base_w = 0.99 * risk_ratio / n if n > 0 else 0.0
        weights = {sym: base_w for sym in final_holdings}
        if risk_ratio < 1.0:
            weights[DEFENSIVE_ETFS[0]] = weights.get(DEFENSIVE_ETFS[0], 0.0) + 0.99 * (1.0 - risk_ratio)

    # ── 市场状态短评 ──
    # 计算沪深300近20日表现
    hs300 = "510300"
    if hs300 in close.columns and len(close) >= 20:
        ret_20d = close[hs300].iloc[-1] / close[hs300].iloc[-20] - 1
        if ret_20d > 0.05:
            market_regime = f"偏强（沪深300近20日 +{ret_20d:.1%}）"
        elif ret_20d < -0.05:
            market_regime = f"偏弱（沪深300近20日 {ret_20d:.1%}）"
        else:
            market_regime = f"震荡（沪深300近20日 {ret_20d:.1%}）"
    else:
        market_regime = "数据不足"

    # ── 操作建议事项 ──
    action_items = []
    if risk_ratio == 0.0:
        action_items.append("* 全部持仓转入防御资产（货币ETF/国债ETF），等待市场企稳")
    elif risk_ratio < 1.0:
        pct_def = int((1 - risk_ratio) * 100)
        action_items.append(f"* 当前处于{pct_def}%防御状态，风险仓位缩减至{int(risk_ratio*100)}%")

    for sym in final_holdings:
        name = ETF_POOL.get(sym, sym)
        w = weights.get(sym, 0)
        val = w * current_capital if w > 0 else 0
        mom_score = all_scores.get(sym, np.nan)
        mom_str = f"{mom_score:.2%}" if not pd.isna(mom_score) else "N/A"
        action_items.append(f"* {name}（{sym}）：目标仓位 {w:.1%}（约 {val:,.0f} 元），动量得分 {mom_str}")

    # ── HMM市场状态补充（可选，不影响主流程）──
    hmm_state_text = ""
    try:
        from market_state import get_latest_market_state
        mkt = get_latest_market_state(close_matrix)
        hmm_state_text = (
            f"HMM状态: {mkt['state'].upper()}, "
            f"bull概率={mkt['bull_prob']:.1%} ({mkt['method']})"
        )
    except Exception:
        hmm_state_text = "HMM状态: 未计算（hmmlearn可能未安装）"

    # ── LLM双Agent决策（hybrid模式）──
    llm_weights = None
    llm_section_lines = []
    if use_llm:
        try:
            from llm_agent import run_dual_agent
            from market_state import get_latest_market_state
            mkt_state = get_latest_market_state(close_matrix)
            logger.info("[Advisor] 调用DeepSeek双Agent生成持仓建议...")
            llm_weights = run_dual_agent(
                close=close_matrix,
                features=features,
                signal_date=last_signal_date,
                market_state=mkt_state,
                top_n=TOP_N,
                fallback_weights=weights,
            )
            logger.info(f"[Advisor] LLM持仓建议: {llm_weights}")
            llm_section_lines = [
                "",
                "── DeepSeek LLM 持仓建议（并列展示）────────",
                "  [保留上方量化建议；以下为双Agent协作生成的LLM建议]",
            ]
            for sym, w in sorted(llm_weights.items(), key=lambda x: -x[1]):
                name = ETF_POOL.get(sym, sym)[:8]
                val = w * current_capital
                llm_section_lines.append(
                    f"  {sym}  {name:<12} {w:.1%}  ≈ {val:,.0f} 元"
                )
        except Exception as e:
            logger.warning(f"[Advisor] LLM建议生成失败（回退到量化）: {e}")
            llm_section_lines = [
                "",
                "── DeepSeek LLM 持仓建议 ─────────────────",
                f"  ⚠️  LLM调用失败: {str(e)[:80]}",
                "  已回退到纯量化建议",
            ]

    # ── 生成完整文字报告 ──
    report_lines = [
        "=" * 62,
        f"  ETF双动量轮动策略 · 当日投资建议",
        f"  生成时间：{today}",
        "=" * 62,
        "",
        f"【市场状态】{market_regime}",
        f"【HMM状态】{hmm_state_text}",
        f"【风控档位】{circuit_status}",
        f"【参考净值】{nav_latest:,.0f} 元",
        f"【当前回撤】{drawdown_pct:.2%}",
        "",
        f"【信号来源】{signal_source}",
        "",
        "── 量化建议持仓（保留）────────────────────────",
    ]
    for item in action_items:
        report_lines.append(item)

    # 插入LLM建议区块（如果有）
    if llm_section_lines:
        report_lines.extend(llm_section_lines)

    report_lines += [
        "",
        "── ETF动量排行（前10名，供参考）─────────────",
        f"  {'排名':<5}{'代码':<8}{'名称':<16}{'动量得分':>10}{'状态':>8}",
        "  " + "-" * 50,
    ]
    for rank, (sym, score) in enumerate(sorted_etfs[:10], 1):
        name = ETF_POOL.get(sym, sym)[:8]
        in_def = "防御" if sym in DEFENSIVE_ETFS else ("持仓" if sym in final_holdings else "")
        report_lines.append(f"  {rank:<5}{sym:<8}{name:<16}{score:>10.2%}{in_def:>8}")

    report_lines += [
        "",
        "── 防御资产得分 ────────────────────────────",
    ]
    for sym, sc in def_scores.items():
        name = ETF_POOL.get(sym, sym)
        sc_str = f"{sc:.2%}" if not pd.isna(sc) else "N/A"
        report_lines.append(f"  {sym}  {name:<16}  得分={sc_str}")

    report_lines += [
        "",
        "── 风险提示 ─────────────────────────────────",
        "  1. 本建议基于历史收盘价，次交易日开盘价执行（T+1）",
        "  2. 动量因子在极端行情（黑天鹅/流动性危机）时可能失效",
        "  3. 跨境ETF涉及汇率风险，请关注人民币汇率走势",
        "  4. 建议仓位仅供参考，实际仓位以自身风险承受能力为准",
        "=" * 62,
    ]

    report_text = "\n".join(report_lines)

    return {
        "date": today,
        "market_regime": market_regime,
        "holdings": final_holdings,
        "weights": weights,
        "scores": all_scores,
        "circuit_status": circuit_status,
        "risk_ratio": risk_ratio,
        "nav_latest": nav_latest,
        "drawdown_pct": drawdown_pct,
        "action_items": action_items,
        "report_text": report_text,
        "llm_weights": llm_weights,
    }


# ─────────────────────────────────────────────
# 调仓前一日建议（含挂单价格）
# ─────────────────────────────────────────────
def generate_pre_rebalance_advice(
    advice: dict,
    close_row: pd.Series,
    execution_date: pd.Timestamp,
    current_holdings: list = None,
    current_weights: dict = None,
    current_capital: float = 1_000_000.0,
) -> str:
    """
    在调仓信号日（执行日前一个交易日）收盘后，
    生成含明日挂单价格的操作建议。

    参数：
    - advice:           generate_advice() 的返回字典
    - close_row:        信号日收盘价 Series
    - execution_date:   明日执行日
    - current_holdings: 当前持仓列表
    - current_weights:  当前各ETF权重 {symbol: weight}
    - current_capital:  当前总资金

    返回：追加了挂单价格表的完整报告文本
    """
    target_weights = advice.get("weights", {})
    if current_weights is None:
        current_weights = {}

    # 计算挂单价
    orders = compute_limit_prices(
        target_weights=target_weights,
        current_weights=current_weights,
        close_row=close_row,
    )

    exec_date_str = execution_date.strftime("%Y-%m-%d")
    signal_date_str = advice.get("date", "")

    lines = [
        advice["report_text"],
        "",
        "=" * 62,
        f"  📋 明日挂单操作指南（{exec_date_str} 开盘前挂单）",
        "=" * 62,
        "",
        f"  信号日期：{signal_date_str}（今日收盘后产生信号）",
        f"  执行日期：{exec_date_str}（明日按挂单价执行）",
        f"  参考资金：{current_capital:,.0f} 元",
        "",
    ]

    if not orders:
        lines.append("  无需操作：目标持仓与当前持仓一致，无调仓需求。")
    else:
        lines.append(f"  {'操作':<6}{'代码':<8}{'名称':<14}{'今收盘':>10}{'挂单价':>10}{'目标仓位':>10}{'变动':>10}{'预计金额':>14}")
        lines.append("  " + "-" * 82)
        for o in orders:
            amount = abs(o["change_pct"]) * current_capital
            lines.append(
                f"  {o['action']:<6}{o['symbol']:<8}{o['name'][:7]:<14}"
                f"{o['close_price']:>10.3f}{o['limit_price']:>10.3f}"
                f"{o['target_weight']:>10.1%}{o['change_pct']:>+10.1%}"
                f"{amount:>14,.0f}"
            )

    lines += [
        "",
        "── 挂单注意事项 ────────────────────────────",
        "  1. 挂单价 = 今日收盘价 ± 0.2%（买入加价/卖出折价以提高成交率）",
        "  2. 建议在开盘前9:15-9:25集合竞价阶段挂单",
        "  3. 若开盘价偏离挂单价超过1%，建议撤单重挂",
        "  4. ETF为T+1交易，当日买入次日方可卖出",
        "=" * 62,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 邮件发送
# ─────────────────────────────────────────────
def send_email_message(
    report_text: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
    email_from: str,
    email_to: str,
    subject: str,
    html_title: str,
) -> bool:
    """
    通过 SMTP 发送投资建议邮件。

    参数：
    - smtp_host  : SMTP服务器，如 "smtp.163.com"、"smtp.gmail.com"
    - smtp_port  : 465=SSL, 587=STARTTLS
    - smtp_user  : 发件邮箱账号
    - smtp_pass  : 邮箱授权码（163/QQ等需开启SMTP服务并获取授权码）
    - email_from : 发件人地址
    - email_to   : 收件人地址（多个用英文逗号分隔）
    - subject    : 邮件主题（默认含日期）

    返回：True=成功，False=失败（详见日志）
    """
    if not all([smtp_host, smtp_user, smtp_pass, email_to]):
        logger.error("邮件发送失败：SMTP 配置不完整，请检查参数")
        return False

    recipients = [addr.strip() for addr in email_to.split(",") if addr.strip()]
    if not recipients:
        logger.error("邮件发送失败：收件人地址为空")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr(("ETF策略顾问", email_from or smtp_user))
    msg["To"] = ", ".join(recipients)

    # 纯文本版本
    msg.attach(MIMEText(report_text, "plain", "utf-8"))

    # HTML 版本（将等宽文本放在 <pre> 中，便于阅读）
    html_body = f"""
    <html><body>
    <h3>{html_title}</h3>
    <pre style="font-family:monospace; font-size:13px; background:#f5f5f5; padding:16px; border-radius:6px;">
{report_text}
    </pre>
    <p style="color:#888; font-size:12px;">本邮件由量化策略系统自动生成，仅供参考，不构成投资建议。</p>
    </body></html>
    """
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        if smtp_port == 465 or smtp_port == 994:
            logger.info(f"使用 SSL 连接 SMTP 服务器 {smtp_host}:{smtp_port}")
            import ssl
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=60) as server:
                server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())
        elif smtp_port == 587:
            logger.info(f"使用 STARTTLS 连接 SMTP 服务器 {smtp_host}:{smtp_port}")
            with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())
        else:
            logger.info(f"使用普通 SMTP 连接服务器 {smtp_host}:{smtp_port}")
            with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
                server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipients, msg.as_string())

        logger.info(f"邮件已发送至：{', '.join(recipients)}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        reason = _classify_smtp_auth_error(e)
        logger.error(f"邮件发送失败：{reason}")
        logger.error(f"SMTP 返回：{e}")
    except smtplib.SMTPServerDisconnected as e:
        logger.error(f"邮件发送失败：{_classify_smtp_disconnect_error(e)}")
    except smtplib.SMTPConnectError:
        logger.error(f"邮件发送失败：无法连接 {smtp_host}:{smtp_port}")
    except Exception as e:
        logger.error(f"邮件发送失败：{e}")
    return False


def send_advice_email(
    report_text: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
    email_from: str,
    email_to: str,
    subject: str = None,
) -> bool:
    """发送投资建议邮件。"""
    if subject is None:
        today = datetime.today().strftime("%Y-%m-%d")
        subject = f"ETF轮动策略 · 投资建议 · {today}"

    return send_email_message(
        report_text=report_text,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_pass=smtp_pass,
        email_from=email_from,
        email_to=email_to,
        subject=subject,
        html_title="ETF双动量轮动策略 · 当日投资建议",
    )


# ─────────────────────────────────────────────
# 独立入口（直接运行 advisor.py）
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="生成当日ETF轮动投资建议")
    parser.add_argument("--capital",   type=float, default=1_000_000.0, help="参考资金量（元）")
    parser.add_argument("--refresh",   action="store_true",             help="强制重新下载数据")
    parser.add_argument(
        "--mode", type=str, default="quant", choices=["quant", "hybrid"],
        help="建议模式: quant=仅量化建议, hybrid=量化建议 + LLM建议"
    )
    parser.add_argument("--smtp-test", action="store_true", help="只测试 SMTP 连接与登录，不运行策略")
    parser.add_argument("--cron-mode", action="store_true", help="cron 模式：仅在今天是调仓信号日时发送邮件")
    # ── 邮件参数（命令行优先，否则从 .env 读取）──
    parser.add_argument("--smtp-host", type=str, default=None, help="SMTP服务器（覆盖.env）")
    parser.add_argument("--smtp-port", type=int, default=None, help="SMTP端口（覆盖.env）")
    parser.add_argument("--smtp-user", type=str, default=None, help="发件邮箱账号（覆盖.env）")
    parser.add_argument("--smtp-pass", type=str, default=None, help="邮箱授权码（覆盖.env）")
    parser.add_argument("--email-from",type=str, default=None, help="发件人地址（覆盖.env）")
    parser.add_argument("--email-to",  type=str, default=None, help="收件人地址（覆盖.env）")
    args = parser.parse_args()

    # 合并邮箱配置：命令行 > .env
    env_email = load_email_config()
    smtp_host  = args.smtp_host  or env_email["smtp_host"]
    smtp_port  = args.smtp_port  or env_email["smtp_port"]
    smtp_user  = args.smtp_user  or env_email["smtp_user"]
    smtp_pass  = args.smtp_pass  or env_email["smtp_pass"]
    email_from = args.email_from or env_email["email_from"]
    email_to   = args.email_to   or env_email["email_to"]

    if args.smtp_test:
        ok = test_smtp_login(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_pass=smtp_pass,
        )
        sys.exit(0 if ok else 1)

    from data_fetcher   import load_data
    from feature_engine import build_features
    from signal_generator import get_rebalance_dates as _get_rebal

    logger.info("加载数据...")
    # advisor 独立运行时：始终拉取到今天，不受回测 END_DATE 限制
    today_str = datetime.today().strftime("%Y-%m-%d")
    import config as cfg
    _orig_end = cfg.END_DATE
    cfg.END_DATE = today_str
    close_matrix, ohlcv_panel = load_data(force_refresh=args.refresh)
    cfg.END_DATE = _orig_end  # 恢复，避免影响同进程其他模块

    logger.info("计算特征...")
    features = build_features(close_matrix, ohlcv_panel)

    # 尝试加载已有净值序列
    nav_df = None
    nav_path = os.path.join(OUTPUT_DIR, "nav_series.csv")
    if os.path.exists(nav_path):
        try:
            nav_df = pd.read_csv(nav_path, index_col=0, parse_dates=True)
        except Exception:
            pass

    advice = generate_advice(
        close_matrix=close_matrix,
        features=features,
        nav_df=nav_df,
        current_capital=args.capital,
        use_llm=(args.mode == "hybrid"),
    )

    # 判断今日是否为调仓信号日
    trading_dates = features["close"].index
    rebal_dates = _get_rebal(trading_dates)
    today_ts = trading_dates[-1]

    if args.cron_mode and not is_latest_trading_day_today(trading_dates):
        logger.info("cron 模式：今天不是最新交易日，跳过发送")
        sys.exit(0)

    is_signal, exec_date = is_pre_rebalance_day(today_ts, trading_dates, rebal_dates)
    if args.cron_mode and not (is_signal and exec_date is not None):
        logger.info("cron 模式：今天不是调仓前一个交易日，跳过发送")
        sys.exit(0)

    if is_signal and exec_date is not None:
        logger.info(f"今日({today_ts.strftime('%Y-%m-%d')})是调仓信号日，"
                    f"明日({exec_date.strftime('%Y-%m-%d')})执行调仓")
        close_row = features["close"].loc[today_ts]
        report_text = generate_pre_rebalance_advice(
            advice=advice,
            close_row=close_row,
            execution_date=exec_date,
            current_capital=args.capital,
        )
    else:
        report_text = advice["report_text"]

    print(report_text)

    if args.cron_mode and not (smtp_host and email_to):
        logger.error("cron 模式要求已配置 SMTP_HOST 和 EMAIL_TO")
        sys.exit(1)

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
