# 中国 ETF 双动量轮动策略

[English Version](README.en.md)

一个面向中国 ETF 市场的量化交易项目，核心目标是在严格控制回撤的前提下，获得长期稳定的正向绝对收益。

项目当前包含两条可运行策略链路：

- 纯量化版本：双引擎 ETF 轮动框架，强调风险控制、仓位约束和可复现回测。
- Hybrid 版本：在量化锚点组合之上，引入 DeepSeek 双 Agent 做受限 overlay 调整，而不是让 LLM 直接重建整个组合。

## 项目特点

- 面向中国市场 ETF：覆盖宽基、行业、风格、QDII、黄金、国债、货币 ETF。
- 双引擎切换：进攻子策略与防守子策略按月切换。
- 多层风控：ATR 风险预算、目标波动率控制、分档熔断、周级风险闸门、常驻防御仓。
- Hybrid 架构：LLM 仅在量化基线附近做小幅倾斜，避免失控换仓。
- 可执行工程结构：包含数据获取、特征工程、信号生成、回测、可视化、投资建议与邮件发送。

## 策略架构

### 1. 纯量化主链

- ETF 池内做截面动量排名。
- 结合绝对动量过滤与防御资产替换。
- 使用 ATR 做风险预算与仓位分配。
- 通过目标波动率和回撤熔断控制组合风险。
- 在进攻与防守引擎之间做月度切换。

### 2. Hybrid 主链

- 先由纯量化策略生成量化锚点持仓与锚点权重。
- Agent 1 负责宏观环境判断。
- Agent 2 只允许在量化锚点附近输出受限乘数，不允许新增资产。
- 系统会自动修复轻微的 JSON 输出问题与单资产超限问题。
- 当 LLM 不可用或信号无效时，会回退到量化基线，而不是输出一套失控组合。

## 项目结构

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

## 环境要求

- Python 3.10+
- Linux / macOS 均可，当前项目主要在 Linux 环境验证

安装依赖：

```bash
cd finance
pip install -r requirements.txt
```

如果要运行 Hybrid 模式，还需要在 `strategy/.env` 中配置：

```env
deepseek_api_key=your_api_key_here
```

## 快速开始

### 运行纯量化回测

```bash
cd finance/strategy
python3 main.py --no-chart
```

### 运行 LLM Hybrid 回测

```bash
cd finance/strategy
python3 main.py --mode hybrid --llm-backtest --no-chart
```

### 生成当日投资建议

纯量化建议：

```bash
cd finance/strategy
python3 advisor.py
```

量化建议 + LLM 建议并列展示：

```bash
cd finance/strategy
python3 advisor.py --mode hybrid
```

### 在主流程里附带生成建议

```bash
cd finance/strategy
python3 main.py --mode hybrid --advice --no-chart
```

## 最近验证结果

以下结果来自当前仓库内已验证的回测记录，区间为 `2015-01-05 ~ 2026-03-31`。

| 版本 | 最终净值 | 年化收益率 | 最大回撤 | 夏普比率 | 交易笔数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 纯量化基线 | 3,787,872 | 12.59% | -8.42% | 1.142 | 913 |
| Hybrid LLM | 3,634,045 | 12.17% | -8.18% | 1.141 | 886 |

Hybrid 版本当前的工程重点不是“让 LLM 取代量化”，而是让 LLM 在量化锚点上做受控增强。因此它更像一个风险受限的 overlay 层，而不是自由裁量型组合生成器。

## 输出内容

回测完成后，结果会写入 `output/` 目录，例如：

- `output/quant_backtest/`
- `output/hybrid_llm_backtest/`

通常会包含：

- 净值序列 CSV
- 交易记录 CSV
- 图表输出（如果未关闭可视化）

## 适合谁

- 想研究中国 ETF 轮动策略的人
- 想把传统量化框架和 LLM Agent 结合的人
- 想从一个可运行、可回测、可扩展的量化项目开始的人

## 风险说明

- 本项目的所有结果均为历史回测，不代表未来收益。
- 中国市场 ETF 的流动性、交易成本、申赎结构和极端行情都会影响真实执行结果。
- Hybrid 模式依赖外部 LLM 服务，存在网络、配额和响应稳定性风险。
- 本项目仅供研究和工程实践，不构成投资建议。