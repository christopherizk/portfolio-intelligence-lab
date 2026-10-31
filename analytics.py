"""Transparent, testable portfolio analytics. Prices must be adjusted for splits/distributions."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index().replace([np.inf, -np.inf], np.nan).dropna(how='any')
    if prices.empty or (prices <= 0).any().any():
        raise ValueError('No valid positive, overlapping adjusted price observations.')
    return prices


def returns_from_prices(prices):
    prices = clean_prices(prices)
    r = prices.pct_change().dropna(how='any')
    if len(r) < 60:
        raise ValueError('At least 60 overlapping daily return observations are required.')
    return r


def portfolio_returns(asset_returns, weights, fee_bps=0, rebalance='Monthly'):
    """Close-to-close returns; beginning-of-day target weights on rebalance dates.

    Fees are deducted on initial deployment and on turnover when rebalancing.
    Returns include an initial deployment fee on the first observed return day.
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Weights must be non-negative and sum to one.')
    r = asset_returns.to_numpy(dtype=float)
    dates = asset_returns.index
    holdings = weights.copy()
    out, turnover = [], []
    for i, day in enumerate(dates):
        trade = 0.0
        if i == 0:
            trade = 1.0  # initial capital deployment; buys sum to 100%
        elif rebalance == 'Daily' or (rebalance == 'Monthly' and day.to_period('M') != dates[i-1].to_period('M')) or (rebalance == 'Quarterly' and day.to_period('Q') != dates[i-1].to_period('Q')):
            trade = float(np.abs(weights - holdings).sum())
            holdings = weights.copy()
        cost = trade * fee_bps / 10000
        gross = float(np.dot(holdings, r[i]))
        net = (1 - cost) * (1 + gross) - 1
        out.append(net)
        turnover.append(trade)
        updated = holdings * (1 + r[i])
        holdings = updated / updated.sum()
    return pd.Series(out, index=dates, name='Portfolio'), pd.Series(turnover, index=dates, name='Turnover')


def equity_curve(r, initial=100_000):
    return initial * (1 + r).cumprod()


def drawdown(r):
    wealth = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)], ignore_index=True)
    return pd.Series(wealth.iloc[1:].to_numpy() / wealth.cummax().iloc[1:].to_numpy() - 1, index=r.index)


def metrics(r, rf=0.0, benchmark=None):
    r = r.dropna()
    n = len(r)
    total = float((1+r).prod()-1)
    annual = float((1+total)**(TRADING_DAYS/n)-1)
    vol = float(r.std(ddof=1)*np.sqrt(TRADING_DAYS))
    excess = r - (1+rf)**(1/TRADING_DAYS)+1
    sharpe = float(excess.mean()/excess.std(ddof=1)*np.sqrt(TRADING_DAYS)) if excess.std(ddof=1)>0 else np.nan
    downside = np.minimum(excess, 0)
    sortino = float(excess.mean()/np.sqrt((downside**2).mean())*np.sqrt(TRADING_DAYS)) if (downside**2).mean()>0 else np.nan
    var95 = float(np.quantile(r, .05))
    cvar95 = float(r[r <= var95].mean())
    d = {'Total return':total,'CAGR':annual,'Volatility':vol,'Sharpe':sharpe,'Sortino':sortino,'Max drawdown':float(drawdown(r).min()),'Historical daily VaR 95%':var95,'Historical daily CVaR 95%':cvar95}
    if benchmark is not None:
        common = pd.concat([r,benchmark],axis=1).dropna()
        if len(common)>2 and common.iloc[:,1].var()>0:
            d['Beta vs benchmark'] = float(common.iloc[:,0].cov(common.iloc[:,1])/common.iloc[:,1].var())
            d['Tracking error'] = float((common.iloc[:,0]-common.iloc[:,1]).std()*np.sqrt(TRADING_DAYS))
    return d


def risk_contributions(returns, weights):
    cov = returns.cov().to_numpy()*TRADING_DAYS
    w = np.asarray(weights,dtype=float)
    v = float(w@cov@w)
    if v <= 0: return np.zeros_like(w)
    return w*(cov@w)/v  # fractional contributions can be negative for hedging assets


def optimize_min_vol(returns):
    n = returns.shape[1]
    cov = returns.cov().to_numpy()*TRADING_DAYS
    result = minimize(lambda w: float(w@cov@w), np.ones(n)/n, method='SLSQP',
                      bounds=[(0,1)]*n, constraints=[{'type':'eq','fun':lambda w: w.sum()-1}],
                      options={'maxiter':2000,'ftol':1e-12})
    if not result.success: raise ValueError(f'Optimization failed: {result.message}')
    return result.x


def stress_pnl(weights, shocks):
    return float(np.dot(np.asarray(weights),np.asarray(shocks)))
