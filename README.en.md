# China ETF Dual Momentum Rotation Strategy

[中文版](README.md)

A quantitative trading project for the China ETF market, designed to pursue long-term positive absolute returns while keeping drawdown under strict control.

The repository currently contains two runnable strategy paths:

- Quant-only version: a dual-engine ETF rotation framework focused on risk control, position constraints, and reproducible backtests.
- Hybrid version: a DeepSeek dual-agent overlay on top of a quantitative anchor portfolio, rather than letting the LLM rebuild the entire portfolio.

## Highlights

- Built for China ETFs: broad index, sector, style, QDII, gold, treasury, and money-market ETFs.
- Dual-engine switching: monthly rotation between offensive and defensive sub-strategies.
- Multi-layer risk control: ATR risk budgeting, target volatility control, tiered circuit breaker, weekly risk gate, and permanent defensive allocation.
- Hybrid architecture: the LLM only tilts around the quantitative baseline with bounded adjustments.
- Executable project structure: data ingestion, feature engineering, signal generation, backtesting, visualization, daily advice, and email delivery.

## Strategy Architecture

### 1. Quant Core Path

- Cross-sectional momentum ranking inside the ETF universe.
- Absolute momentum filtering with defensive asset substitution.
- ATR-based risk budgeting and position sizing.
- Target volatility and drawdown circuit breaker for portfolio-level risk control.
- Monthly switching between offensive and defensive engines.

### 2. Hybrid Path

- The quant strategy first generates anchor holdings and anchor weights.
- Agent 1 handles macro regime judgment.
- Agent 2 is only allowed to output bounded multipliers around the quant anchor, with no new assets allowed.
- The system automatically repairs minor JSON formatting issues and single-name cap violations.
- When the LLM is unavailable or the response is invalid, the system falls back to the quant baseline instead of producing an uncontrolled portfolio.

## Project Structure

```text
finance/
├── README.md
├── README.en.md
├── requirements.txt
├── data/
├── output/
└── strategy/
    ├── main.py
    ├── advisor.py
    ├── config.py
    ├── data_fetcher.py
    ├── feature_engine.py
    ├── signal_generator.py
    ├── risk_manager.py
    ├── backtest_engine.py
    ├── performance.py
    ├── visualization.py
    ├── market_state.py
    └── llm_agent.py
```

## Requirements

- Python 3.10+
- Linux or macOS; the current project has been primarily validated on Linux

Install dependencies:

```bash
cd finance
pip install -r requirements.txt
```

To run the Hybrid mode, add the following to `strategy/.env`:

```env
deepseek_api_key=your_api_key_here
```

## Quick Start

### Run the quant-only backtest

```bash
cd finance/strategy
python3 main.py --no-chart
```

### Run the Hybrid LLM backtest

```bash
cd finance/strategy
python3 main.py --mode hybrid --llm-backtest --no-chart
```

### Generate daily strategy advice

Quant-only advice:

```bash
cd finance/strategy
python3 advisor.py
```

Quant advice plus side-by-side LLM advice:

```bash
cd finance/strategy
python3 advisor.py --mode hybrid
```

### Generate advice inside the main workflow

```bash
cd finance/strategy
python3 main.py --mode hybrid --advice --no-chart
```

## Latest Validated Results

The following figures come from validated backtest records in this repository over `2015-01-05 ~ 2026-03-31`.

| Version | Final NAV | Annual Return | Max Drawdown | Sharpe Ratio | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quant baseline | 3,787,872 | 12.59% | -8.42% | 1.142 | 913 |
| Hybrid LLM | 3,634,045 | 12.17% | -8.18% | 1.141 | 886 |

The current goal of the Hybrid version is not to let the LLM replace the quant engine. It is designed as a controlled overlay layer on top of a quantitative anchor portfolio.

## Outputs

Backtest outputs are written to the `output/` directory, for example:

- `output/quant_backtest/`
- `output/hybrid_llm_backtest/`

Typical output files include:

- NAV series CSV
- trade log CSV
- chart exports when visualization is enabled

## Who This Is For

- Researchers exploring China ETF rotation strategies
- Developers combining traditional quant workflows with LLM agents
- Anyone who wants a runnable, backtestable, and extensible quant research project

## Risk Notice

- All results in this repository are historical backtests and do not guarantee future performance.
- Real-world execution can be affected by liquidity, fees, ETF structure, and extreme market conditions in the China market.
- The Hybrid mode depends on an external LLM service and therefore carries network, quota, and response stability risk.
- This repository is for research and engineering purposes only and is not investment advice.