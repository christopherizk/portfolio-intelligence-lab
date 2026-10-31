"""Portfolio Intelligence Lab | run: streamlit run app.py"""
from datetime import date, timedelta
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analytics import (returns_from_prices, portfolio_returns, equity_curve, drawdown,
                       metrics, risk_contributions, optimize_min_vol, stress_pnl)

st.set_page_config(page_title='Portfolio Intelligence Lab', page_icon='◈', layout='wide')
st.markdown("""<style>
/* Institutional black interface, inspired by professional trading platforms. */
:root {color-scheme:dark;}
html,body,[data-testid="stAppViewContainer"],.stApp {background:#050505!important;color:#F2F2F2!important;}
[data-testid="stHeader"] {background:#050505!important;}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {background:#101010!important;border-right:1px solid #303030!important;}
[data-testid="stSidebar"] * {color:#E6E6E6!important;}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {color:#C7C7C7!important;}
h1,h2,h3,h4,p,span,label,div[data-testid="stMarkdownContainer"] {color:inherit;}
h1,h2,h3 {color:#F5F5F5!important;letter-spacing:-.025em;}
h1 {font-weight:700!important;}
[data-testid="stCaptionContainer"],.stCaption {color:#A5A5A5!important;}
[data-testid="stMetric"] {background:#151515!important;border:1px solid #303030!important;padding:18px!important;border-radius:8px!important;}
[data-testid="stMetricLabel"] * {color:#AAAAAA!important;}
[data-testid="stMetricValue"] * {color:#F5F5F5!important;font-size:clamp(1.5rem,2.3vw,2.15rem)!important;letter-spacing:-.035em!important;}
[data-testid="stMetricValue"] {overflow:visible!important;}
[data-testid="stMetric"] {min-width:0!important;}
[data-testid="stMetricLabel"] {min-height:1.6rem;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {border-bottom:1px solid #333!important;gap:12px!important;}
[data-testid="stTabs"] button[role="tab"] {color:#BEBEBE!important;background:transparent!important;}
[data-testid="stTabs"] button[role="tab"] * {color:inherit!important;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color:#FFFFFF!important;border-bottom:2px solid #F0F0F0!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background:#F0F0F0!important;}
[data-baseweb="input"] > div,[data-baseweb="select"] > div,[data-baseweb="popover"] {background:#1A1A1A!important;border-color:#3C3C3C!important;color:#FFFFFF!important;}
input,textarea,[data-baseweb="select"] input {color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;}
[data-testid="stNumberInput"] button {background:#202020!important;color:#FFFFFF!important;}
[data-testid="stNumberInput"] button svg {fill:#FFFFFF!important;}
[data-testid="stDataFrame"], [data-testid="stTable"] {border:1px solid #333!important;}
[data-testid="stAlert"] {background:#191919!important;color:#E5E5E5!important;}
[data-testid="stFileUploader"] section {background:#161616!important;border-color:#444!important;}
[data-testid="stDownloadButton"] button {background:#202020!important;color:#FFFFFF!important;border:1px solid #555!important;}
hr {border-color:#333!important;}
</style>""", unsafe_allow_html=True)

# Keep all charts consistent with the terminal's dark, high-contrast palette.
CHART_BG = '#111111'
CHART_FG = '#E9E9E9'
def style_chart(fig, height=None):
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color=CHART_FG, family='Arial, sans-serif', size=12),
        margin=dict(l=25,r=20,t=35,b=35),
        legend=dict(font=dict(color=CHART_FG)),
        xaxis=dict(gridcolor='#2C2C2C',zerolinecolor='#2C2C2C',tickfont=dict(color='#BFBFBF')),
        yaxis=dict(gridcolor='#2C2C2C',zerolinecolor='#2C2C2C',tickfont=dict(color='#BFBFBF')),
        **({'height':height} if height else {})
    )
    return fig

ASSETS = {'SPY':'US Equities · S&P 500','VGK':'European Equities','AGG':'US Investment-Grade Bonds',
          'GLD':'Gold','BIL':'US Treasury Bills / Cash Proxy'}
TICKERS = list(ASSETS)
DEFAULT = [35,20,25,10,10]
COLORS = ['#F2F2F2','#8E969F','#5AADB0','#B8B8B8','#636363']

@st.cache_data(ttl=3600,show_spinner=False)
def download_prices(tickers,start,end):
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, threads=False, group_by='column')
    if raw.empty: raise ValueError('Yahoo Finance returned no price history.')
    if isinstance(raw.columns,pd.MultiIndex):
        p = raw['Close']
    else:
        p = raw[['Close']].rename(columns={'Close':tickers[0]})
    p = p.reindex(columns=tickers)
    return p

st.markdown('**INDEPENDENT QUANTITATIVE RESEARCH**  ·  MULTI-ASSET PORTFOLIO')
st.title('Portfolio Intelligence Lab')
st.caption('Performance analytics · Risk attribution · Portfolio optimization · Stress testing | USD-denominated, illustrative portfolio')

with st.sidebar:
    st.header('Portfolio controls')
    source=st.radio('Price data', ['Yahoo Finance (live download)','Upload adjusted-close CSV'], help='Internet required for Yahoo Finance. Upload a CSV with Date, SPY, VGK, AGG, GLD, BIL columns if offline.')
    end=st.date_input('End date (exclusive)',value=date.today()+timedelta(days=1))
    start=st.date_input('Start date',value=date.today()-timedelta(days=365*3))
    capital=st.number_input('Initial notional (USD)',min_value=1000,value=100000,step=10000)
    st.subheader('Target allocation')
    allocations={t:st.number_input(f'{ASSETS[t]} ({t}) %',min_value=0.0,max_value=100.0,value=float(DEFAULT[i]),step=5.0) for i,t in enumerate(TICKERS)}
    st.caption(f"Total allocation: {sum(allocations.values()):.1f}% (must equal 100%)")
    rebalance=st.selectbox('Rebalancing', ['Monthly','Quarterly','Daily','Buy & hold'])
    fees=st.number_input('Transaction cost per $ traded (basis points)',0.0,100.0,5.0,1.0)
    rf=st.number_input('Assumed annual risk-free rate (%)',0.0,20.0,3.0,0.25)/100
    uploaded=st.file_uploader('Upload price data (CSV)',type=['csv'],help='Optional alternative to Yahoo Finance. Required columns: Date, SPY, VGK, AGG, GLD, BIL.') if source.startswith('Upload') else None

if abs(sum(allocations.values())-100)>1e-7:
    st.error('Target weights must sum to exactly 100%. Adjust the sidebar allocation.'); st.stop()
if start>=end:
    st.error('Start date must precede end date.'); st.stop()
if source.startswith('Upload') and uploaded is None:
    st.info('Upload a CSV of **adjusted closing prices** with columns: Date, SPY, VGK, AGG, GLD, BIL. The app never fabricates historical prices.'); st.stop()
try:
    if uploaded is not None:
        prices=pd.read_csv(uploaded,parse_dates=['Date']).set_index('Date').reindex(columns=TICKERS)
    else:
        with st.spinner('Downloading adjusted historical prices…'):
            prices=download_prices(TICKERS,start.isoformat(),end.isoformat())
    prices.index=pd.to_datetime(prices.index).tz_localize(None) if prices.index.tz is not None else pd.to_datetime(prices.index)
    prices=prices.loc[(prices.index.date>=start)&(prices.index.date<end)]
    missing=[t for t in TICKERS if prices[t].dropna().empty]
    if missing: raise ValueError(f'Missing price history: {", ".join(missing)}')
    asset_returns=returns_from_prices(prices)
    prices=prices.loc[asset_returns.index.min()-pd.Timedelta(days=7):asset_returns.index.max()]
except Exception as exc:
    st.error(f'Unable to load valid data: {exc}')
    st.info('Check internet access, dates, or upload a CSV of adjusted closing prices.'); st.stop()

w=np.array([allocations[t]/100 for t in TICKERS])
pr,turnover=portfolio_returns(asset_returns,w,fees,rebalance)
benchmark=asset_returns['SPY'].rename('SPY Benchmark')
stat=metrics(pr,rf,benchmark)
benchmark_curve=equity_curve(benchmark,capital)
curve=equity_curve(pr,capital)

st.caption(f'Observed data: {asset_returns.index.min():%d %b %Y} – {asset_returns.index.max():%d %b %Y} · {len(asset_returns):,} daily returns · Adjusted close · All instruments priced in USD · Source: {"User-uploaded CSV" if uploaded is not None else "Yahoo Finance via yfinance"}')

# Two balanced rows prevent Streamlit's metric ellipsis at laptop widths.
p1,p2,p3=st.columns(3)
p1.metric('Portfolio value',f'${curve.iloc[-1]:,.0f}')
p2.metric('Total return',f'{stat["Total return"]:+.2%}')
p3.metric('Annualized volatility',f'{stat["Volatility"]:.2%}')
p4,p5=st.columns(2)
p4.metric('Sharpe ratio',f'{stat["Sharpe"]:.2f}')
p5.metric('Maximum drawdown',f'{stat["Max drawdown"]:.2%}')

performance,risk,construction,stress,method=st.tabs(['Performance','Risk & attribution','Optimization','Stress testing','Methodology & export'])
with performance:
    st.subheader('Portfolio vs. S&P 500 benchmark')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=curve.index,y=curve.values,name='Portfolio (net of fees)',line=dict(color='#F2F2F2',width=3)))
    fig.add_trace(go.Scatter(x=benchmark_curve.index,y=benchmark_curve.values,name='SPY benchmark (gross)',line=dict(color='#8E969F',width=2)))
    fig.update_layout(yaxis_title='Value (USD)',legend_orientation='h'); style_chart(fig,410)
    st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        st.subheader('Portfolio drawdown')
        fig=px.area(x=drawdown(pr).index,y=drawdown(pr).values,labels={'x':'Date','y':'Drawdown'},color_discrete_sequence=['#8E969F'])
        style_chart(fig,310)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.subheader('Calendar-month returns')
        monthly=(1+pr).resample('ME').prod()-1
        st.dataframe(monthly.rename_axis('Month').to_frame('Monthly return').rename_axis(index=None).set_axis(monthly.index.strftime('%b %Y')).style.format({'Monthly return':'{:.2%}'}),use_container_width=True,height=320)
    st.caption('SPY benchmark is buy-and-hold and excludes trading fees; the portfolio includes assumed fees. Comparison is therefore not fee-identical.')
with risk:
    c1,c2=st.columns(2)
    with c1:
        st.subheader('Asset allocation')
        fig=px.pie(values=w,names=[f'{t} · {ASSETS[t]}' for t in TICKERS],color_discrete_sequence=COLORS,hole=.57)
        style_chart(fig,350)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.subheader('Variance risk contribution')
        contributions=risk_contributions(asset_returns,w)
        fig=px.bar(x=TICKERS,y=contributions*100,labels={'x':'ETF','y':'Contribution to portfolio variance (%)'},color=TICKERS,color_discrete_sequence=COLORS)
        fig.update_layout(showlegend=False); style_chart(fig,350)
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Daily return correlations')
    corr=asset_returns.corr()
    fig=px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu',aspect='auto')
    style_chart(fig,400)
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Risk statistics')
    st.dataframe(pd.DataFrame({'Metric':list(stat),'Value':[f'{v:.3f}' if 'Sharpe' in k or 'Sortino' in k or 'Beta' in k else f'{v:.2%}' for k,v in stat.items()]}),hide_index=True,use_container_width=True)
    st.caption('VaR/CVaR are historical one-day return tail estimates, not guaranteed future losses. Risk contributions are covariance-based and may be negative.')
with construction:
    st.subheader('Long-only minimum-volatility allocation')
    st.write('Optimization uses historical daily return covariance, full-sample estimates and weights constrained between 0% and 100%. **In-sample illustration, not a validated out-of-sample strategy.**')
    try:
        opt=optimize_min_vol(asset_returns)
        st.dataframe(pd.DataFrame({'ETF':TICKERS,'Current target':w,'Minimum-volatility target':opt}).style.format({'Current target':'{:.1%}','Minimum-volatility target':'{:.1%}'}),hide_index=True,use_container_width=True)
        vol_current=np.sqrt(w@asset_returns.cov().to_numpy()@w*252)
        vol_opt=np.sqrt(opt@asset_returns.cov().to_numpy()@opt*252)
        a,b=st.columns(2)
        a.metric('Current ex-ante annual volatility',f'{vol_current:.2%}')
        b.metric('Optimized ex-ante annual volatility',f'{vol_opt:.2%}')
        st.warning('Do not interpret in-sample optimized weights as a reliable forward-looking recommendation. Historical covariances change.')
    except Exception as e: st.error(f'Optimizer unavailable: {e}')
with stress:
    st.subheader('Custom one-period price shock')
    st.write('Enter **hypothetical simultaneous ETF price changes**. This is a linear portfolio shock, not a macroeconomic forecasting model.')
    cols=st.columns(len(TICKERS))
    defaults=[-20,-18,-5,10,0]
    shocks=np.array([cols[i].number_input(f'{t} shock (%)',-100.0,200.0,float(defaults[i]),1.0,key=f'shock_{t}')/100 for i,t in enumerate(TICKERS)])
    pnl=stress_pnl(w,shocks)
    a,b=st.columns(2)
    a.metric('Hypothetical portfolio shock',f'{pnl:+.2%}')
    b.metric('Notional P&L',f'${capital*pnl:+,.0f}')
    st.dataframe(pd.DataFrame({'ETF':TICKERS,'Target weight':w,'Hypothetical shock':shocks,'P&L contribution':w*shocks}).style.format({'Target weight':'{:.1%}','Hypothetical shock':'{:+.1%}','P&L contribution':'{:+.2%}'}),hide_index=True,use_container_width=True)
    st.caption('No second-order effects, intra-period rebalancing, slippage, financing costs or nonlinear exposure modeled.')
with method:
    st.subheader('Research methodology')
    st.caption('Transparent assumptions, historical inputs and reproducible calculations. All results are hypothetical.')
    st.markdown('''
- **Investable proxies:** SPY (US equities), VGK (European equities), AGG (US investment-grade bonds), GLD (gold), BIL (short Treasury bills/cash proxy). All are USD-traded ETFs; BIL is **not** risk-free cash.
- **Data:** Yahoo Finance adjusted closing prices through `yfinance`, or a user-supplied adjusted-close CSV. Prices include corporate-action adjustments as provided by the source; data can be revised or missing.
- **Portfolio mechanics:** Daily simple returns; target-weight rebalancing at the beginning of the first trading day of each selected period. Between rebalances, weights drift with asset returns. Fees = absolute dollar turnover × user-defined bps, including initial investment. No leverage, shorting, taxes, FX conversion or bid–ask modeling beyond the assumed bps.
- **Performance:** Geometric compounded return; CAGR uses 252 trading days; volatility and Sharpe use 252 trading days; risk-free annual rate converted to an effective daily rate.
- **Risk:** Historical 5th-percentile one-day VaR, average worst-5%-tail return (CVaR), peak-to-trough maximum drawdown; variance risk contributions = `wᵢ(Σw)ᵢ/(wᵀΣw)`.
- **Optimization:** Historical full-sample covariance, long-only, fully invested, no turnover constraint. **In-sample only**, susceptible to estimation error and look-ahead if interpreted as backtested performance.
- **Benchmark:** SPY buy-and-hold, gross of fees; not a matched multi-asset benchmark.
- **Limitations:** ETF total-return proxies do not perfectly represent direct index ownership; market close times differ, historical returns do not predict future performance, and stress scenarios are user assumptions.
''')
    st.divider()
    st.subheader('Data & reproducibility')
    st.write('Export the underlying daily calculations or the adjusted closing prices to inspect the analysis independently. These downloads do not change the portfolio or its assumptions.')
    export=asset_returns.copy()
    export['Portfolio return net of fees']=pr
    export['SPY benchmark return']=benchmark
    export['Portfolio value USD']=curve
    export['SPY benchmark value USD']=benchmark_curve
    export['Rebalance turnover']=turnover
    dl1,dl2=st.columns(2)
    with dl1:
        st.download_button('Export daily calculations · CSV',export.to_csv(index_label='Date').encode(),file_name='portfolio_intelligence_daily_calculations.csv',mime='text/csv',use_container_width=True)
    with dl2:
        st.download_button('Export adjusted prices · CSV',prices.to_csv(index_label='Date').encode(),file_name='portfolio_intelligence_adjusted_prices.csv',mime='text/csv',use_container_width=True)
    st.caption('For research and education only. Not financial advice. Project results are hypothetical and do not represent actual trading.')
