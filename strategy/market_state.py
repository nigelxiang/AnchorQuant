"""
market_state.py — HMM 市场状态识别

使用 2状态 Gaussian HMM 对沪深300日收益率序列建模，判断宏观市场环境：
  - State "bull" : 低波动上涨（牛市/震荡期）
  - State "bear" : 高波动下跌（熊市/危机期）

设计原则：
1. 滚动训练窗口（HMM_TRAIN_DAYS）：每次只用历史数据训练，严防未来泄漏。
2. 输出两列：
   - "hmm_state"      : str（"bull" / "bear"）
   - "hmm_bull_prob"  : float（处于bull状态的概率）
3. 依赖 hmmlearn（可选）：若未安装，自动降级为波动率阈值法判断。

安装：
    pip install hmmlearn scikit-learn
"""

import logging
import numpy as np
import pandas as pd

from config import HMM_N_STATES, HMM_TRAIN_DAYS, BENCHMARK_ETF

logger = logging.getLogger(__name__)

# 尝试导入 hmmlearn，未安装时降级
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    logger.warning(
        "hmmlearn 未安装，市场状态识别将降级为波动率阈值法。"
        "安装命令：pip install hmmlearn"
    )


# ─────────────────────────────────────────────
# 降级方案：波动率阈值法
# ─────────────────────────────────────────────
def _vol_threshold_state(
    log_ret: np.ndarray,
    vol_window: int = 20,
    vol_threshold: float = 0.20,
) -> tuple:
    """
    简单降级方案：年化波动率 > vol_threshold → "bear"，否则 "bull"。
    返回：(state_label str, bull_prob float)
    """
    if len(log_ret) < vol_window:
        return "bull", 0.6

    recent_vol = np.std(log_ret[-vol_window:]) * np.sqrt(252)
    recent_ret = np.mean(log_ret[-vol_window:]) * 252

    # 高波动 + 下跌 → bear
    if recent_vol > vol_threshold and recent_ret < 0:
        bull_prob = max(0.05, 0.5 - (recent_vol - vol_threshold) * 2)
        return "bear", round(bull_prob, 3)
    elif recent_vol > vol_threshold:
        bull_prob = 0.5
        return "neutral", round(bull_prob, 3)
    else:
        bull_prob = min(0.95, 0.5 + (vol_threshold - recent_vol) * 2)
        return "bull", round(bull_prob, 3)


# ─────────────────────────────────────────────
# 核心：滚动HMM推断（防未来泄漏版）
# ─────────────────────────────────────────────
def _fit_and_decode_hmm(
    log_ret_series: np.ndarray,
    n_states: int = 2,
    n_iter: int = 50,
) -> tuple:
    """
    对传入的收益率序列拟合 GaussianHMM 并解码最后一个状态。
    
    返回：(state_label str, bull_prob float)
    状态命名规则：均值更高的状态 = "bull"，均值更低的状态 = "bear"。
    """
    if len(log_ret_series) < n_states * 10:
        return "bull", 0.6

    X = log_ret_series.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42,
    )
    try:
        model.fit(X)
    except Exception as e:
        logger.debug(f"HMM fit 失败: {e}")
        return "bull", 0.6

    # 获取所有状态的对数概率（前向算法后验）
    try:
        _, posteriors = model.score_samples(X)  # (T, n_states)
    except Exception:
        try:
            hidden_states = model.predict(X)
            last_state = hidden_states[-1]
        except Exception:
            return "bull", 0.6
        # 无法获取概率时用均值排序
        means = model.means_.flatten()
        bull_state = int(np.argmax(means))
        label = "bull" if last_state == bull_state else "bear"
        return label, 0.7 if label == "bull" else 0.3

    # 用状态均值排名：均值高 = bull
    means = model.means_.flatten()
    bull_state = int(np.argmax(means))
    bear_state = int(np.argmin(means))

    last_posteriors = posteriors[-1]  # shape (n_states,)
    bull_prob = float(last_posteriors[bull_state])

    if bull_prob >= 0.5:
        label = "bull"
    else:
        label = "bear"

    return label, round(bull_prob, 3)


# ─────────────────────────────────────────────
# 公开接口：计算全历史的市场状态序列
# ─────────────────────────────────────────────
def compute_market_states(
    close: pd.DataFrame,
    benchmark: str = None,
    train_days: int = None,
    n_states: int = None,
) -> pd.DataFrame:
    """
    对基准ETF（默认沪深300）的历史收益率序列，
    使用滚动 HMM 推断每个调仓日的市场状态。

    参数：
    - close:      宽表收盘价 DataFrame（date x symbol）
    - benchmark:  基准ETF代码（默认 config.BENCHMARK_ETF）
    - train_days: 训练窗口（默认 config.HMM_TRAIN_DAYS）
    - n_states:   HMM状态数（默认 config.HMM_N_STATES）

    返回：
        DataFrame with columns ["hmm_state", "hmm_bull_prob"]，index=日期
        每行代表"当日（信号日）的市场状态"，基于历史数据推断，无未来泄漏。
    """
    if benchmark is None:
        benchmark = BENCHMARK_ETF
    if train_days is None:
        train_days = HMM_TRAIN_DAYS
    if n_states is None:
        n_states = HMM_N_STATES

    if benchmark not in close.columns:
        logger.warning(f"基准ETF {benchmark} 不在 close 中，使用等权平均收益代替")
        price_series = close.mean(axis=1)
    else:
        price_series = close[benchmark]

    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    dates = log_ret.index

    state_labels = []
    bull_probs = []

    for i, dt in enumerate(dates):
        # 只使用截至当前日期（含）的历史数据，严格防未来泄漏
        available = log_ret.loc[:dt].values
        if len(available) < train_days // 2:
            # 数据不足：默认bull
            state_labels.append("bull")
            bull_probs.append(0.6)
            continue

        window_data = available[-train_days:]

        if _HMM_AVAILABLE:
            label, prob = _fit_and_decode_hmm(window_data, n_states=n_states)
        else:
            label, prob = _vol_threshold_state(window_data)

        state_labels.append(label)
        bull_probs.append(prob)

    result = pd.DataFrame(
        {"hmm_state": state_labels, "hmm_bull_prob": bull_probs},
        index=dates,
    )

    logger.info(
        f"市场状态计算完成: "
        f"bull={sum(s=='bull' for s in state_labels)}, "
        f"bear={sum(s=='bear' for s in state_labels)}, "
        f"方法={'HMM' if _HMM_AVAILABLE else '波动率阈值'}"
    )

    return result


# ─────────────────────────────────────────────
# 快速查询：获取最新市场状态（实盘/建议使用）
# ─────────────────────────────────────────────
def get_latest_market_state(
    close: pd.DataFrame,
    benchmark: str = None,
    train_days: int = None,
) -> dict:
    """
    只计算最新一个时间点的市场状态，用于实盘决策。
    比 compute_market_states() 快（不滚动计算全历史）。

    返回：{"state": str, "bull_prob": float, "method": str}
    """
    if benchmark is None:
        benchmark = BENCHMARK_ETF
    if train_days is None:
        train_days = HMM_TRAIN_DAYS

    if benchmark not in close.columns:
        price_series = close.mean(axis=1)
    else:
        price_series = close[benchmark]

    log_ret = np.log(price_series / price_series.shift(1)).dropna().values
    window_data = log_ret[-train_days:]

    method = "HMM" if _HMM_AVAILABLE else "波动率阈值"

    if _HMM_AVAILABLE:
        label, prob = _fit_and_decode_hmm(window_data)
    else:
        label, prob = _vol_threshold_state(window_data)

    return {"state": label, "bull_prob": prob, "method": method}


# ─────────────────────────────────────────────
# 序列化为 LLM-friendly 文本（供 Agent Prompt 使用）
# ─────────────────────────────────────────────
def format_market_state_for_llm(
    state_dict: dict,
    style_ratios_row: pd.Series = None,
    cross_border_row: pd.Series = None,
) -> str:
    """
    将市场状态数据格式化为简洁的自然语言描述，
    作为 DeepSeek Agent 1 (宏观分析师) 的 Prompt 输入片段。
    """
    state = state_dict.get("state", "unknown")
    prob = state_dict.get("bull_prob", 0.5)
    method = state_dict.get("method", "未知")

    lines = [f"HMM市场状态: {state.upper()} (bull概率={prob:.1%}, 方法={method})"]

    if style_ratios_row is not None and not style_ratios_row.empty:
        lines.append("A股风格比值（截至信号日）:")
        for name, val in style_ratios_row.dropna().items():
            lines.append(f"  {name} = {val:.4f}")

    if cross_border_row is not None and not cross_border_row.empty:
        lines.append("跨境市场近20日涨跌:")
        for sym_col, val in cross_border_row.dropna().items():
            sym = sym_col.replace("xborder_", "")
            lines.append(f"  {sym}: {val:+.2%}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    logging.basicConfig(level=logging.INFO)

    from data_fetcher import load_data

    print("加载数据...")
    close_mtx, _ = load_data()

    print("计算最新市场状态...")
    state = get_latest_market_state(close_mtx)
    print(f"  状态: {state['state']}, bull概率: {state['bull_prob']:.1%}, 方法: {state['method']}")
