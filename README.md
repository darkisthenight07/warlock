# Crypto RL Trader Pipeline

A highly modular and structured framework designed for building, testing, and verifying Reinforcement Learning (RL) agents for Bitcoin/Crypto trading. The project features a custom Gymnasium-compliant market environment, an advanced portfolio simulator with realistic market friction modeling, and a highly customizable feature engineering suite.

## Project State & Functionality

The repository contains the complete foundational infrastructure required to train an RL trading agent. All internal components—including data pipelines, indicator creation, order execution simulation, and reward mechanics—are complete and validated via standalone test suites. 

### Core Modules

1. **Data Management (`src/data_manager/`)**
   * **Downloader & Cleaner:** Automates downloading historical OHLCV data from exchanges (e.g., Binance) via `downloader.py`.
   * **Anomaly Detection:** Tracks market structural anomalies such as extreme wick deviations via rolling windows and wick multipliers.

2. **Feature Engineering Suite (`src/features/`)**
   * Implements a pipeline creating over 35 distinct features across 5 major categories: Price Action, Candlestick structures, Momentum, Volatility, and Volume indicators.
   * Includes automated feature profiling, producing visualization graphs (`graphs/features/`) to diagnose correlation profiles, rolling Sharpe ratios, trend strength, and distribution patterns.

3. **Custom Gymnasium Environment (`src/env/`)**
   * `gym_bitcoin.py` provides a custom Gymnasium interface designed to pass price tensors and historical lookback windows seamlessly to standard RL networks.

4. **Advanced Portfolio Simulator (`src/portfolio/`)**
   * Emulates live execution constraints including separate maker/taker fee rates, minimum trade national cutoffs, and a fixed basis-points (`fixed_bps`) slippage model.
   * Manages trade ledger tracking, rebalancing step delta thresholds to prevent dust trades, and strict automated risk rules (e.g., maximum drawdown liquidations).

5. **Asymmetric Reward Engineering (`src/env/rewards.py`)**
   * Computes objective functions driven by custom reward criteria.
   * Features dynamic rolling Sharpe ratio buffers combined with configurable scale penalties for drawdowns and high-frequency overtrading.

---

## Configuration

All modules are completely decentralized and governed by `config.yaml`. Adjusting this file allows you to instantly alter exchange parameters, swap active technical indicators, fine-tune trading fees, configure lookback observation windows, or tweak reward structures without touching core code.

---

## Getting Started

### Prerequisites
* Python 3.8+
* `TA-Lib` C-library dependencies (required for technical market indicators)

### 1. Installation

Clone this repository and set up your local development environment:

```bash
# Clone the repository
git clone https://github.com/darkisthenight07/warlock
cd crypto-rl-trader

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```


## Running the Data & Feature Pipeline
To invoke the pipeline orchestration engine (which downloads data, builds clean feature matrices, prints dataset details, and builds diagnostic asset charts in your local graphs directory):
```bash
python main.py

```

## Component Verification Tests
The repository packages separate verification testing scripts to guarantee that your custom gym environment, simulated account portfolios, and mathematical reward components operate within perfect limits. Run them via the following scripts:

```bash
# Verify reward scaling properties, buffer mechanics, and penalties
python test_rewards.py

# Verify trade execution flows, fee charges, slippage models, and liquidations
python test_portfolio.py

# Verify Gymnasium state handling, lookback observations, step updates, and resets
python test_env.py

```
