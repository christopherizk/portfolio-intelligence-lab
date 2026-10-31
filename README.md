# Portfolio Intelligence Lab

A reproducible, interactive multi-asset portfolio research dashboard built with Python and Streamlit.

## Run locally

Requires Python 3.10+ and an internet connection for Yahoo Finance downloads.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

If Yahoo Finance is unavailable, select **Upload adjusted-close CSV** and upload a CSV with `Date,SPY,VGK,AGG,GLD,BIL` columns, containing authentic adjusted closing prices (not made-up returns). Date should be in YYYY-MM-DD format.

## What it does

- Target allocations and periodic rebalancing, with configurable transaction-cost bps
- Compounded performance vs SPY, drawdown and monthly returns
- Sharpe, Sortino, historical daily VaR and CVaR, beta and tracking error
- Covariance-based risk contribution and ETF correlation heatmap
- Long-only minimum-volatility optimization (clearly labeled **in-sample**)
- User-defined price-shock scenarios and notional P&L
- CSV exports of adjusted prices and all daily portfolio calculations

## Important assumptions

All ETF proxies trade in USD. Initial $100,000 is illustrative, not real money. BIL is a Treasury-bill ETF rather than cash. Yahoo Finance is an unofficial market data provider; data availability and historical adjustments can change. Trading fees are modeled as basis points of absolute traded dollar notional, including initial deployment. Benchmark SPY is gross of fees, while the portfolio is net of assumed fees. Optimization is **not** out-of-sample. See the Methodology tab for formulas and limitations.

## Tests

```bash
python -m unittest discover -s tests -v
```

## LinkedIn (only after running and reviewing results)

**Portfolio Intelligence Lab — Multi-Asset Analytics & Risk Dashboard**

Built an interactive Python dashboard to analyze a hypothetical multi-asset ETF portfolio, measure risk-adjusted performance, attribute portfolio risk, and simulate market stress scenarios. Implemented configurable rebalancing, transaction costs, and long-only minimum-volatility optimization using historical adjusted prices.
