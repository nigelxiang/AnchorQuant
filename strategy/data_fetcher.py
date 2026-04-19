"""
data_fetcher.py — 数据获取与ETL清洗
数据源优先级（由 config.DATA_SOURCE 控制）：
  'sina'    : AkShare fund_etf_hist_sina — 新浪资讯后端，无频率限制（默认）
  'em'      : AkShare fund_etf_hist_em   — 东方财富后端，有限速风险
  'auto'    : Sina 失败则自动降级到 em

- 本地 pickle 缓存，避免重复请求
- ETL：对齐交易日历、前向填充、异常值标记、缺失值处理
"""

import os
import time
import pickle
import random
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import akshare as ak

from config import (
    ETF_POOL, START_DATE, END_DATE,
    DATA_DIR, BENCHMARK_ETF
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 缓存有效期（秒）
CACHE_TTL = 86400          # 整体缓存有效期：1天
PER_ETF_CACHE_TTL = 7 * 86400  # 单只ETF缓存有效期：7天

# 数据源优先级（可在 config.py 中通过 DATA_SOURCE 覆盖）
# "akshare"  : 仅用 AkShare
# "baostock" : 仅用 BaoStock
# "auto"     : AkShare 失败后自动降级到 BaoStock
DATA_SOURCE = getattr(__import__('config'), 'DATA_SOURCE', 'sina')


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Sina 接口（主用：无频率限制）
# ─────────────────────────────────────────────
def _sina_code(symbol: str) -> str:
    """将6位ETF代码转为 Sina 格式 (sh510300 / sz159915)"""
    sh_prefixes = ('50', '51', '52', '11', '13')
    return ('sh' if symbol[:2] in sh_prefixes else 'sz') + symbol


def _apply_split_correction(df: pd.DataFrame, max_single_day_drop: float = 0.20) -> pd.DataFrame:
    """
    对 Sina 原始（不复权）价格序列做前复权（qfq）修正。

    原理：A股ETF每日涨跌停限制≤±15%。若单日收盘价跌幅 > max_single_day_drop（默认20%），
    则认定为基金分拆（基金份额增加，单位净值等比下降）事件，而非真实市场损失。
    将该日之前的所有价格等比例缩小（乘以 close[t] / close[t-1]），
    使价格序列连续可比（等价于前复权处理），消除分拆造成的回测收益失真。

    threshold 说明：
    - A股ETF正常日限 ±10%，QDII放宽至±15%，均不触发20%门槛
    - 基金分拆通常为2:1、5:1等，价格单日下跌 50%+，清晰可识别
    """
    if df.empty or "close" not in df.columns:
        return df

    df = df.copy()
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    close = df["close"].values.astype(float)
    n = len(close)

    # 累积调整因子数组（从旧到新），初始全1
    adj = np.ones(n, dtype=float)
    events = []

    for i in range(1, n):
        if close[i - 1] > 0 and close[i] > 0:
            factor = close[i] / close[i - 1]
            if factor < (1.0 - max_single_day_drop):
                # 检测到分拆：历史所有价格 × factor（缩小到分拆后的单位价格水平）
                adj[:i] *= factor
                events.append((df.index[i].date(), f"{1/factor:.2f}:1 分拆检测"))

    if events:
        logger.info(f"[split_correction] 检测到价格分拆事件: {events}")

    # 应用调整因子到所有价格列
    for col in price_cols:
        df[col] = df[col].values * adj

    return df



def _fetch_sina(
    symbol: str,
    start: str,
    end: str,
    retries: int = 3,
) -> pd.DataFrame:
    """
    AkShare fund_etf_hist_sina — 新浪资讯后端，无频率限制。
    下载原始（不复权）日线数据，并自动应用分拆修正（前复权效果），
    确保价格序列连续可比，动量因子和回测收益计算正确。
    注：Sina 返回列名已是英文 (date/open/high/low/close/volume/amount)。
    """
    code = _sina_code(symbol)
    for attempt in range(retries):
        try:
            df = ak.fund_etf_hist_sina(symbol=code)
            break
        except Exception as e:
            logger.warning(f"[Sina][{symbol}] 第{attempt+1}次失败: {e}")
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1) + random.uniform(0, 1))
            else:
                return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[start:end].copy()

    if df.empty:
        logger.warning(f"[Sina][{symbol}] 截断后为空，检查日期区间 {start}~{end}")
        return pd.DataFrame()

    # 统一列名（Sina 返回 amount，重命名为 turnover）
    if "amount" in df.columns:
        df.rename(columns={"amount": "turnover"}, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 分拆修正（前复权）：消除基金份额分拆导致的价格断层
    df = _apply_split_correction(df)

    logger.info(f"[Sina][{symbol}] 下载成功，{len(df)} 行")
    return df


# ─────────────────────────────────────────────
# 单只ETF下载
# ─────────────────────────────────────────────
def _per_etf_cache_path(symbol: str, start: str, end: str) -> str:
    """单只ETF的独立缓存文件路径"""
    return os.path.join(DATA_DIR, f"{symbol}_{start}_{end}.pkl")


def _fetch_single_etf(
    symbol: str,
    start: str,
    end: str,
    retries: int = 5,
    base_delay: float = 5.0,
    use_per_etf_cache: bool = True,
) -> pd.DataFrame:
    """
    下载单只ETF日线数据。

    数据源策略（由 DATA_SOURCE 控制）：
    - 'sina' : AkShare fund_etf_hist_sina，Sina 资讯后端，无频率限制（默认）
    - 'em'   : AkShare fund_etf_hist_em，东方财富后端（有限速风险）
    - 'auto' : 先尝试 Sina，失败后降级到 em
    """
    # 单只ETF缓存
    if use_per_etf_cache:
        cache_path = _per_etf_cache_path(symbol, start, end)
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < PER_ETF_CACHE_TTL:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if not cached.empty:
                    return cached

    # ── em模式：直接尝试东方财富 ──
    if DATA_SOURCE == 'em':
        df = _fetch_akshare_em(symbol, start, end, retries, base_delay)
        if not df.empty:
            _save_per_etf_cache(symbol, start, end, df)
        return df

    # ── sina 模式：Sina 返回不复权价格，仅作后备 ──
    df = _fetch_sina(symbol, start, end)

    # ── auto 降级：Sina 失败则尝试 em（qfq复权）──
    if df.empty and DATA_SOURCE == 'auto':
        logger.info(f"[{symbol}] Sina 失败，切换至 em(qfq) ...")
        df = _fetch_akshare_em(symbol, start, end, retries, base_delay)

    if not df.empty:
        _save_per_etf_cache(symbol, start, end, df)
    return df


def _save_per_etf_cache(symbol: str, start: str, end: str, df: pd.DataFrame):
    cache_path = _per_etf_cache_path(symbol, start, end)
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)


def _fetch_akshare_em(
    symbol: str,
    start: str,
    end: str,
    retries: int = 5,
    base_delay: float = 5.0,
) -> pd.DataFrame:
    """
    东方财富后端（限速风险，已包含指数退避重试）。
    使用 adjust="qfq"（前复权）：所有历史价格调整到当前分红/拆分基准，
    保证价格序列连续可比，动量因子和回测收益计算均正确。
    """
    df = pd.DataFrame()
    for attempt in range(retries):
        try:
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
            break
        except Exception as e:
            logger.warning(f"[em][{symbol}] 第{attempt+1}次请求失败: {e}")
            if attempt < retries - 1:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 2)
                logger.info(f"[em][{symbol}] 等待 {wait:.1f}s 后重试...")
                time.sleep(wait)
            else:
                return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "turnover",
        "涨跌幅": "pct_chg", "涨跌额": "chg",
    }
    df.rename(columns=col_map, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    keep = [c for c in ["open", "high", "low", "close", "volume", "turnover", "pct_chg"] if c in df.columns]
    df = df[keep].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"[em][{symbol}] 下载成功，{len(df)} 行")
    return df


# ─────────────────────────────────────────────
# 批量下载（含本地缓存）
# ─────────────────────────────────────────────
def fetch_all_etfs(
    etf_pool: dict = None,
    start: str = START_DATE,
    end: str = END_DATE,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    批量下载ETF池数据，返回 {symbol: DataFrame}。
    自动读写pickle缓存，缓存超期或 force_refresh=True 时重新拉取。
    """
    if etf_pool is None:
        etf_pool = ETF_POOL

    cache_file = os.path.join(DATA_DIR, f"etf_cache_{start}_{end}.pkl")

    if not force_refresh and os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < CACHE_TTL:
            logger.info(f"读取本地缓存: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    result = {}
    total = len(etf_pool)
    for i, (symbol, name) in enumerate(etf_pool.items(), 1):
        logger.info(f"[{i}/{total}] 下载 {symbol} {name} ...")
        df = _fetch_single_etf(symbol, start, end)
        if not df.empty:
            result[symbol] = df
            # 每成功下载一只即写入整体缓存（断点保护）
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

        # 请求间隔：1.2s基础 + 随机抖动0~0.8s，避免触发东方财富限速
        if i < total:   # 最后一只不需要等待
            sleep_sec = 1.2 + random.uniform(0, 0.8)
            time.sleep(sleep_sec)

    # 持久化缓存
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"数据缓存: {cache_file}")

    return result


# ─────────────────────────────────────────────
# ETL：对齐、清洗、特征预处理
# ─────────────────────────────────────────────
def build_price_matrix(
    raw_data: dict[str, pd.DataFrame],
    col: str = "close",
) -> pd.DataFrame:
    """
    将各ETF的单列（默认close）拼接为宽表，对齐交易日历。
    - 对齐到最完整的交易日历（取并集）
    - 前向填充非交易日（停牌/节假日）
    - 至多连续填充5日（>5日视为异常，保留NaN）
    """
    frames = {sym: df[col].rename(sym) for sym, df in raw_data.items() if col in df.columns}
    price_matrix = pd.concat(frames, axis=1)
    price_matrix.index = pd.to_datetime(price_matrix.index)
    price_matrix.sort_index(inplace=True)

    # 前向填充，上限5日（模拟停牌/节假日）
    price_matrix = price_matrix.ffill(limit=5)

    return price_matrix


def remove_outliers(df: pd.DataFrame, max_daily_move: float = 0.15) -> pd.DataFrame:
    """
    对复权后日收益率做异常值检测：
    - 单日涨跌幅 > ±15% 的点标记为NaN（ETF理论上不应触及涨跌停，极端值为数据错误）
    - 用前向填充修复
    注意：入参为价格宽表，返回清洗后价格宽表。
    """
    daily_ret = df.pct_change()
    mask = daily_ret.abs() > max_daily_move
    df_clean = df.copy()
    # 异常价格点还原为NaN后前向填充
    df_clean[mask] = np.nan
    df_clean = df_clean.ffill(limit=3)
    outlier_count = mask.sum().sum()
    if outlier_count > 0:
        logger.warning(f"移除异常价格点 {outlier_count} 处（单日涨跌幅>±{max_daily_move*100:.0f}%）")
    return df_clean


def build_ohlcv_panel(
    raw_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    构建各只ETF对齐后的OHLCV字典（用于ATR计算）。
    """
    # 获取统一的交易日历（以close宽表的index为基准）
    close_matrix = build_price_matrix(raw_data, "close")
    date_index = close_matrix.index

    panel = {}
    for sym, df in raw_data.items():
        df_aligned = df.reindex(date_index).ffill(limit=5)
        panel[sym] = df_aligned
    return panel


# ─────────────────────────────────────────────
# 数据完整性检查
# ─────────────────────────────────────────────
def validate_data(price_matrix: pd.DataFrame, min_history_days: int = 252) -> pd.DataFrame:
    """
    检查并丢弃历史数据不足 min_history_days 的ETF。
    """
    valid_count = price_matrix.notna().sum()
    drop_cols = valid_count[valid_count < min_history_days].index.tolist()
    if drop_cols:
        logger.warning(f"历史数据不足{min_history_days}日，剔除: {drop_cols}")
        price_matrix = price_matrix.drop(columns=drop_cols)
    return price_matrix


# ─────────────────────────────────────────────
# 主函数（独立运行时的快速验证）
# ─────────────────────────────────────────────
def load_data(force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    完整ETL流程入口，返回：
    - close_matrix: 清洗后的收盘价宽表 (date x symbol)
    - ohlcv_panel:  各ETF对齐OHLCV字典
    """
    raw = fetch_all_etfs(force_refresh=force_refresh)
    if not raw:
        raise RuntimeError("数据下载失败，请检查网络或AkShare版本")

    close_matrix = build_price_matrix(raw, "close")
    close_matrix = remove_outliers(close_matrix)
    close_matrix = validate_data(close_matrix)
    ohlcv_panel  = build_ohlcv_panel(raw)

    logger.info(
        f"数据加载完成: {len(close_matrix)}个交易日 × {len(close_matrix.columns)}只ETF "
        f"({close_matrix.index[0].date()} ~ {close_matrix.index[-1].date()})"
    )
    return close_matrix, ohlcv_panel


if __name__ == "__main__":
    close_mtx, ohlcv = load_data(force_refresh=False)
    print(close_mtx.tail())
    print(f"\n各ETF数据起始日期:")
    print(close_mtx.apply(lambda c: c.first_valid_index()))
