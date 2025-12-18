import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Halal Crisis Lab | The Skeptic's Guide", 
    layout="wide", 
    page_icon="🛡️"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-label { font-size: 0.85rem; color: #666; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #2c3e50; }
    .metric-sub { font-size: 0.85rem; margin-top: 5px; }
    .highlight-box {
        background-color: #e8f6f3;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #d1f2eb;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. EDUCATIONAL NARRATIVE ENGINE ---
SCENARIO_CONTEXT = {
    "Full History (2019-Present)": {
        "dates": ("2019-12-20", "TODAY"),
        "story": "The Long View. Covers the Covid crash, 2021 bubble, 2022 inflation shock, and 2023 AI boom.",
        "bias": "Neutral",
        "default_inf": 3.0
    },
    "The COVID Crash (2020-2021)": {
        "dates": ("2020-01-01", "2021-12-31"),
        "story": "A 'Stay at Home' economy. Tech giants soared while physical banks suffered. Halal funds often shine here.",
        "bias": "Bullish for Halal",
        "default_inf": 1.5
    },
    "Inflation Bear Market (2022)": {
        "dates": ("2022-01-01", "2022-12-31"),
        "story": "The Skeptic's validation. Rates spiked. Banks profited; Tech crashed. Halal funds (0% Banks, High Tech) feel the pain.",
        "bias": "Bearish for Halal",
        "default_inf": 8.0
    },
    "The AI Boom (2023-Present)": {
        "dates": ("2023-01-01", "TODAY"),
        "story": "The return of Tech. The 'Magnificent 7' drove gains. Halal funds captured this upside due to high Tech exposure.",
        "bias": "Bullish for Halal",
        "default_inf": 3.5
    }
}

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Simulation Lab")

# A. Scenario Selection
st.sidebar.subheader("1. Choose Your Crisis")
period_name = st.sidebar.selectbox("Market Scenario", list(SCENARIO_CONTEXT.keys()))
current_ctx = SCENARIO_CONTEXT[period_name]

# B. Asset Configuration
with st.sidebar.expander("2. Asset & Currency", expanded=True):
    halal_ticker = st.selectbox("Halal Fund", ["HLAL (Wahed FTSE)", "SPUS (SP Funds S&P 500)"])
    h_sym = halal_ticker.split()[0]
    
    currency_mode = st.radio("Base Currency", ["USD ($) - Original", "GBP (£) - UK Investor"])
    curr_sym = "£" if "GBP" in currency_mode else "$"

# C. The "Skeptic" Toggles
with st.sidebar.expander("3. Reality Checks (Advanced)", expanded=True):
    st.caption("Apply real-world friction to the simulation.")
    inflation_input = st.slider(
        "Inflation Rate (Annual %)", 
        min_value=0.0, max_value=15.0, 
        value=current_ctx['default_inf'], 
        step=0.5,
        help=f"Simulates purchasing power loss. Default is set to {current_ctx['default_inf']}% for this period."
    )
    inflation_rate = inflation_input / 100

    apply_fees = st.checkbox(
        "Apply 'Cost of Conscience' (Fees + Purification)", 
        value=False, 
        help="Simulates a 0.55% total drag."
    )

# D. Strategy Logic
with st.sidebar.expander("4. Investment Strategy", expanded=False):
    strategy = st.radio("Method", ["Lump Sum", "DCA (Monthly)"])
    initial_cap = st.number_input(f"Initial Capital ({curr_sym})", 1000, 1000000, 10000, step=1000)
    monthly_add = st.number_input(f"Monthly Add ({curr_sym})", 0, 10000, 500, step=100) if "DCA" in strategy else 0

# --- 4. DATA ENGINE (CLEAN & ROBUST) ---
@st.cache_data
def fetch_data(scenario, h_ticker, use_gbp):
    """
    Fetches Market Data using standard yfinance with error handling.
    """
    start_str, end_str = SCENARIO_CONTEXT[scenario]["dates"]
    if end_str == "TODAY":
        end_str = datetime.now().strftime('%Y-%m-%d')
    
    tickers = ['SPY', h_ticker, '^TNX', 'XLK'] 
    
    try:
        # Standard download - let yfinance handle the session internally
        df = yf.download(tickers, start=start_str, end=end_str, progress=False, auto_adjust=False)
        
        # 1. Handle Empty Download (Rate Limit or Connection)
        if df.empty:
            st.warning(f"⚠️ Yahoo Finance returned no data. You might be Rate Limited. Please wait 15 minutes and refresh.")
            return None # Return None instead of crashing

        # 2. Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df['Adj Close']
            except KeyError:
                try:
                    df = df['Close']
                except KeyError:
                    df = df.droplevel(0, axis=1)
        
        # 3. Fill basic gaps
        df = df.ffill()

        # 4. GBP Logic (SAFE MERGE)
        if use_gbp:
            fx = yf.download("GBPUSD=X", start=start_str, end=end_str, progress=False)
            
            if isinstance(fx.columns, pd.MultiIndex):
                fx = fx['Adj Close'] if 'Adj Close' in fx.columns else fx['Close']
            else:
                fx = fx['Adj Close'] if 'Adj Close' in fx else fx['Close']
            
            if fx.empty:
                 st.warning("⚠️ FX Data missing. Defaulting to USD (1.0).")
                 df['FX'] = 1.0
            else:
                fx.name = "FX"
                df = df.join(fx, how='left')
                df['FX'] = df['FX'].ffill().fillna(1.0)
                
                for col in ['SPY', h_ticker, 'XLK']:
                    if col in df.columns:
                        df[col] = df[col] / df['FX']
        
        # 5. Risk Free Rate
        if '^TNX' in df.columns:
            df['RiskFree_Daily'] = (df['^TNX'].ffill().fillna(4.0) / 100) / 252
        else:
            df['RiskFree_Daily'] = 0.04 / 252
        
        # 6. Final Clean
        df = df.dropna(subset=['SPY', h_ticker])
        
        return df

    except Exception as e:
        st.error(f"Connection Error: {e}")
        st.stop()

# Load Data
df_market = fetch_data(period_name, h_sym, "GBP" in currency_mode)

# --- SAFETY CHECK ---
if df_market is None or df_market.empty or len(df_market) < 2:
    st.error(f"⚠️ Insufficient data. This is likely a Yahoo Finance Rate Limit (Too many refreshes). Please stop refreshing and wait 15-30 minutes for the block to lift.")
    st.stop()

# --- 5. CALCULATION ENGINE ---
def run_simulation(data, initial, monthly, is_dca, h_ticker, inflation_r, apply_fees):
    sim = data.copy()
    
    # Drag Factors
    inf_drag = inflation_r / 252
    fee_drag = (0.0055 / 252) if apply_fees else 0 
    
    total_h_drag = inf_drag + fee_drag
    total_s_drag = inf_drag 
    
    # Wealth Simulation
    sim['Cash_Invested'] = initial
    
    if not is_dca:
        sim['Norm_H'] = sim[h_ticker] / sim[h_ticker].iloc[0]
        sim['Norm_S'] = sim['SPY'] / sim['SPY'].iloc[0]
        
        sim['Drag_H'] = (1 - total_h_drag) ** np.arange(len(sim))
        sim['Drag_S'] = (1 - total_s_drag) ** np.arange(len(sim))
        
        sim[f'Port_{h_ticker}'] = sim['Norm_H'] * initial * sim['Drag_H']
        sim['Port_SPY'] = sim['Norm_S'] * initial * sim['Drag_S']
        
    else:
        sim['Month'] = sim.index.to_period('M')
        monthly_flows = sim.groupby('Month').head(1).index
        
        sim['Units_H'] = 0.0
        sim['Units_S'] = 0.0
        curr_units_h, curr_units_s = 0.0, 0.0
        curr_inv = initial
        
        # Initial Buy
        curr_units_h += initial / sim.iloc[0][h_ticker]
        curr_units_s += initial / sim.iloc[0]['SPY']
        
        for date, row in sim.iterrows():
            if date in monthly_flows and date != sim.index[0]:
                curr_units_h += monthly / row[h_ticker]
                curr_units_s += monthly / row['SPY']
                curr_inv += monthly
            
            sim.at[date, 'Units_H'] = curr_units_h
            sim.at[date, 'Units_S'] = curr_units_s
            sim.at[date, 'Cash_Invested'] = curr_inv
            
        sim['Drag_H'] = (1 - total_h_drag) ** np.arange(len(sim))
        sim['Drag_S'] = (1 - total_s_drag) ** np.arange(len(sim))
        
        sim[f'Port_{h_ticker}'] = (sim['Units_H'] * sim[h_ticker]) * sim['Drag_H']
        sim['Port_SPY'] = (sim['Units_S'] * sim['SPY']) * sim['Drag_S']

    # Returns
    sim['Ret_H'] = sim[f'Port_{h_ticker}'].pct_change()
    sim['Ret_S'] = sim['Port_SPY'].pct_change()
    
    return sim

df_sim = run_simulation(df_market, initial_cap, monthly_add, "DCA" in strategy, h_sym, inflation_rate, apply_fees)

# --- 6. METRIC CALCULATIONS ---
final_h = df_sim[f'Port_{h_sym}'].iloc[-1]
final_s = df_sim['Port_SPY'].iloc[-1]
invested = df_sim['Cash_Invested'].iloc[-1]

ret_h = (final_h / invested) - 1
ret_s = (final_s / invested) - 1

# Max Drawdown
dd_h_series = (df_sim[f'Port_{h_sym}'] / df_sim[f'Port_{h_sym}'].cummax()) - 1
max_dd_h = dd_h_series.min()

dd_s_series = (df_sim['Port_SPY'] / df_sim['Port_SPY'].cummax()) - 1
max_dd_s = dd_s_series.min()

# Risk Metrics
def calc_risk_metrics(series, risk_free):
    excess = series - risk_free
    sharpe = (excess.mean() / excess.std()) * (252**0.5) if excess.std() != 0 else 0
    downside = excess[excess < 0]
    sortino = (excess.mean() / downside.std()) * (252**0.5) if downside.std() != 0 else 0
    return sharpe, sortino

sharpe_h, sortino_h = calc_risk_metrics(df_sim['Ret_H'], df_market['RiskFree_Daily'])
sharpe_s, sortino_s = calc_risk_metrics(df_sim['Ret_S'], df_market['RiskFree_Daily'])


# --- 7. DASHBOARD ---
st.title(f"🛡️ Halal Crisis Lab: {period_name}")

st.markdown(f"""
<div class="highlight-box">
    <h5>The Context: {current_ctx['story']}</h5>
    <p><b>Structural Bias:</b> {current_ctx['bias']} | <b>Simulated Inflation:</b> {inflation_rate*100:.1f}%</p>
</div>
""", unsafe_allow_html=True)

# Metrics
c1, c2, c3, c4, c5 = st.columns(5)

def metric_html(label, value, sub, color, tooltip):
    return f"""
    <div class="metric-box" title="{tooltip}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub" style="color: {color}">{sub}</div>
    </div>
    """

with c1:
    st.markdown(metric_html("Invested Capital", f"{curr_sym}{invested:,.0f}", "Cash Basis", "black", "Total cash put into the system."), unsafe_allow_html=True)
with c2:
    col = "#2ecc71" if ret_h > 0 else "#e74c3c"
    lbl = "Real Return" if inflation_rate > 0 else "Total Return"
    st.markdown(metric_html(f"Halal ({h_sym})", f"{curr_sym}{final_h:,.0f}", f"{ret_h:+.2%} {lbl}", col, "Final portfolio value after strategy."), unsafe_allow_html=True)
with c3:
    col = "#2ecc71" if ret_s > 0 else "#e74c3c"
    st.markdown(metric_html("S&P 500", f"{curr_sym}{final_s:,.0f}", f"{ret_s:+.2%} {lbl}", col, "Benchmark comparison."), unsafe_allow_html=True)
with c4:
    col = "#e74c3c" if max_dd_h < -0.20 else "black"
    st.markdown(metric_html("Max Pain (Drawdown)", f"{max_dd_h:.1%}", f"vs Mrkt: {max_dd_s:.1%}", col, "The single worst drop from the peak value."), unsafe_allow_html=True)
with c5:
    col = "#2ecc71" if sortino_h > sortino_s else "#e74c3c"
    st.markdown(metric_html("Risk Efficiency", f"{sortino_h:.2f}", f"Sharpe: {sharpe_h:.2f}", col, "Main: Sortino (Downside Risk). Sub: Sharpe (Total Risk)."), unsafe_allow_html=True)

st.markdown("---")

# Visuals
tab1, tab2, tab3 = st.tabs(["📈 Wealth Curve & Tech Correlation", "🧠 The Pro's Desk (Risk Deep Dive)", "🧬 Sector Skew"])

with tab1:
    st.markdown("#### Portfolio Growth vs. Technology Sector")
    show_tech = st.checkbox("Overlay Tech Sector (XLK) - See the correlation", value=False)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sim.index, y=df_sim[f'Port_{h_sym}'], name=f'Halal ({h_sym})', line=dict(color='#2ecc71', width=3)))
    fig.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Port_SPY'], name='S&P 500', line=dict(color='gray', width=2)))
    
    if show_tech:
        norm_xlk = (df_market['XLK'] / df_market['XLK'].iloc[0]) * initial_cap
        fig.add_trace(go.Scatter(x=df_market.index, y=norm_xlk, name='Tech (XLK) Proxy', line=dict(color='#9b59b6', dash='dot', width=2), opacity=0.7))

    fig.add_trace(go.Scatter(x=df_sim.index, y=df_sim['Cash_Invested'], name='Cash Invested', line=dict(color='blue', dash='dash', width=1)))
    y_label = "Real Value (Purchasing Power)" if inflation_rate > 0 else f"Portfolio Value ({curr_sym})"
    fig.update_layout(yaxis_title=y_label, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### 🧠 Advanced Risk Analysis")
    risk_data = pd.DataFrame({
        "Metric": ["Sharpe Ratio (Total Volatility)", "Sortino Ratio (Downside Risk)", "Max Drawdown"],
        f"Halal ({h_sym})": [f"{sharpe_h:.2f}", f"{sortino_h:.2f}", f"{max_dd_h:.1%}"],
        "S&P 500": [f"{sharpe_s:.2f}", f"{sortino_s:.2f}", f"{max_dd_s:.1%}"],
    })
    st.table(risk_data.set_index("Metric"))

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("The 'Sector Skew' Explanation")
        skew_data = pd.DataFrame({
            "Sector": ["Technology", "Healthcare", "Financials (Banks)", "Energy/Other"],
            "Halal_Weight": [45, 20, 0, 35],
            "SP500_Weight": [28, 13, 13, 46]
        })
        fig_skew = go.Figure(data=[
            go.Bar(name='Halal Fund', x=skew_data['Sector'], y=skew_data['Halal_Weight'], marker_color='#2ecc71'),
            go.Bar(name='S&P 500', x=skew_data['Sector'], y=skew_data['SP500_Weight'], marker_color='gray')
        ])
        fig_skew.update_layout(barmode='group')
        st.plotly_chart(fig_skew, use_container_width=True)
    with c2:
        st.info("Because Islamic Finance prohibits interest (Riba), these funds cannot hold Financials. The capital naturally flows into Technology to fill the gap.")

with st.expander("🎓 Professor's Commentary: How to read this analysis", expanded=False):
    st.markdown(f"**The Inflation Trap:** You set inflation to **{inflation_rate*100:.1f}%**. If 'Real Return' is negative, you lost wealth in reality.")
