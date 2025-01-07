import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from datetime import datetime, timedelta
import io


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}     
            footer {visibility: hidden;}       
            div[data-testid="stSidebar"] {
                display: none;                
            }
            </style>
            """
st.set_page_config(layout="wide", page_title="Analyse de Portefeuille")
st.markdown(hide_st_style, unsafe_allow_html=True)


default_portfolio = {
    'AEM':   (50,  63.87),
    'AKAM':  (50,  96.82),
    'AMZN':  (50,  170.96),
    '0R37.IL': (35, 434.01),
    'BLK':   (5,   1070),
    'CCJ':   (900, 39.76),
    'LEU':   (150, 85.64),
    'FCX':   (100, 49.21),
    'HPE':   (100, 20.24),
    'INTC':  (400, 23.77),
    'LMT':   (35,  521.45),
    'MSFT':  (65,  366.76),
    'SMR':   (0,   18.95),
    'PLTR.VI': (200, 15.03),
    'RTX':   (85,  105),
    'CRM':   (15,  251.12),
    'ABB.ST':(65,  35.11),
    'BCLM.F':(0,   1432.10),
    'GALDZ.XC':(50,82.75),
    'LZAGY': (5,   533),
    'SZLMY': (15,  714.74),
    'UBS':   (850, 23.50),
    'BAESY': (900, 11),
    'SHEL':  (150, 24.11)
}


real_names = {
    'AEM':      'Agnico Eagle Mines',
    'AKAM':     'Akamai Technologies',
    'AMZN':     'Amazon.com, Inc.',
    '0R37.IL':  'UNKNOWN (0R37.IL)',
    'BLK':      'BlackRock, Inc.',
    'CCJ':      'Cameco Corporation',
    'LEU':      'Centrus Energy Corp.',
    'FCX':      'Freeport-McMoRan',
    'HPE':      'Hewlett Packard Enterprise',
    'INTC':     'Intel Corporation',
    'LMT':      'Lockheed Martin Corp.',
    'MSFT':     'Microsoft Corporation',
    'SMR':      'NuScale Power Corp. (SMR?)',
    'PLTR.VI':  'Palantir (Vienna)',
    'RTX':      'RTX Corp. (Raytheon)',
    'CRM':      'Salesforce, Inc.',
    'ABB.ST':   'ABB Ltd (Sweden)',
    'BCLM.F':   'BCLM (Frankfurt)',
    'GALDZ.XC': 'UNKNOWN (GALDZ.XC)',
    'LZAGY':    'Linde PLC (ADR)',
    'SZLMY':    'UNKNOWN (SZLMY)',
    'UBS':      'UBS Group AG',
    'BAESY':    'BAE Systems PLC (ADR)',
    'SHEL':     'Shell PLC',
    'SPY':      'SPDR S&P 500 ETF',
    'QQQ':      'Invesco QQQ Trust',
    'DIA':      'SPDR Dow Jones Industrial',
    'IWM':      'iShares Russell 2000',
    'VTI':      'Vanguard Total Stock Market ETF'
}


def fetch_current_price(ticker: str) -> float:
    try:
        df = yf.download(ticker, period='1d', progress=False)
        if not df.empty:
            return df['Close'][-1]
    except:
        pass
    return 0.0

def fetch_industry(ticker_symbol: str) -> str:
    try:
        t_info = yf.Ticker(ticker_symbol).info
        return t_info.get("industry", "Unknown")
    except:
        return "Unknown"

def fetch_country(ticker_symbol: str) -> str:
    try:
        t_info = yf.Ticker(ticker_symbol).info
        return t_info.get("country", "Inconnu")
    except:
        return "Inconnu"
    
def get_ticker_from_isin(isin):

    try:
        ticker_info = yf.Ticker(isin)
        return ticker_info.info['symbol'] 
    except KeyError:
        return 'n/a'
    except Exception as e:
        return 'n/a' 

def build_portfolio_df(list_stocks: dict, weighting_mode: str='market_value') -> pd.DataFrame:
    data = []
    for ticker, (shares, purchase_price) in list_stocks.items():
        cp = 0.0
        mv = 0.0
        if shares > 0:
            cp = fetch_current_price(ticker)
            mv = cp * shares
        data.append((ticker, shares, cp, mv))
    
    df = pd.DataFrame(data, columns=['TICKER','SHARES','CURRENT_PRICE','MARKET_VALUE'])
    if weighting_mode=='market_value':
        total_mv = df['MARKET_VALUE'].sum()
        df['WEIGHT'] = np.where(total_mv>0, df['MARKET_VALUE']/total_mv, 0.0)
    else:
        total_sh = df['SHARES'].sum()
        df['WEIGHT'] = np.where(total_sh>0, df['SHARES']/total_sh, 0.0)
    return df

def get_real_name(ticker: str) -> str:
    return real_names.get(ticker, ticker)

def get_historical_data(tickers, years=1):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years*365)
    df_all = pd.DataFrame()
    for t in tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, progress=False)
            if not df.empty:
                df_all[t] = df['Close']
        except:
            pass
    df_all = df_all.ffill().dropna(how='all', axis=1)
    return df_all

def generate_excel_download(df: pd.DataFrame, file_name: str="portfolio.xlsx") -> bytes:
    """
    Takes a pandas DataFrame and returns the bytes for an Excel file.
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Portfolio")
    
    output.seek(0)
    return output.getvalue()

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free=0.0):
    p_ret = np.sum(mean_returns * weights)*252
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return p_ret, p_vol

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free=0.0):
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix, risk_free)
    return -( (p_ret - risk_free)/p_vol ) if p_vol>0 else 999

def maximize_sharpe(mean_returns, cov_matrix, risk_free=0.0, bound=(0,1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free)
    constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1})
    bounds = tuple([bound]*num_assets)
    result = minimize(
        negative_sharpe_ratio,
        x0=[1./num_assets]*num_assets,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def minimize_volatility(mean_returns, cov_matrix, bound=(0,1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type':'eq','fun': lambda x: np.sum(x)-1})
    bounds = tuple([bound]*num_assets)
    result = minimize(
        portfolio_volatility,
        x0=[1./num_assets]*num_assets,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

def portfolio_return(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

def efficient_opt(mean_returns, cov_matrix, return_target, bound=(0,1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = (
        {'type':'eq','fun': lambda x: portfolio_return(x, mean_returns, cov_matrix) - return_target},
        {'type':'eq','fun': lambda x: np.sum(x)-1}
    )
    bounds = tuple([bound]*num_assets)
    result = minimize(
        portfolio_volatility,
        x0=[1./num_assets]*num_assets,
        args=args,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    return result

def calculate_efficient_frontier(mean_returns, cov_matrix, risk_free=0.0, bound=(0,1)):
    # Max Sharpe
    max_sr_res = maximize_sharpe(mean_returns, cov_matrix, risk_free, bound)
    max_sr_w = max_sr_res['x']
    max_sr_r, max_sr_vol = portfolio_performance(max_sr_w, mean_returns, cov_matrix)

    # Min Vol
    min_vol_res = minimize_volatility(mean_returns, cov_matrix, bound)
    min_vol_w = min_vol_res['x']
    min_vol_r, min_vol_vol = portfolio_performance(min_vol_w, mean_returns, cov_matrix)

    target_returns = np.linspace(min_vol_r, max_sr_r, 20)
    frontier_vols = []
    for tr in target_returns:
        eff_res = efficient_opt(mean_returns, cov_matrix, tr, bound)
        frontier_vols.append(eff_res['fun'])
    
    return (max_sr_r, max_sr_vol, max_sr_w,
            min_vol_r, min_vol_vol, min_vol_w,
            target_returns, frontier_vols)

def build_efficient_frontier_figure(
        mean_returns,
        cov_matrix,
        user_weights=None,
        risk_free=0.0,
        bound=(0,1),
        bench_return=None,
        bench_vol=None,
        bench_name="Index"
    ):
    (max_sr_r, max_sr_vol, max_sr_w,
     min_vol_r, min_vol_vol, min_vol_w,
     target_returns, frontier_vols) = calculate_efficient_frontier(mean_returns, cov_matrix, risk_free, bound)
    
    max_sr_r *= 100; max_sr_vol *= 100
    min_vol_r *= 100; min_vol_vol *= 100
    frontier_ret = [tr*100 for tr in target_returns]
    frontier_vol = [fv*100 for fv in frontier_vols]

    # EF
    trace_ef = go.Scatter(
        x=frontier_vol,
        y=frontier_ret,
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(dash='dash', color='blue')
    )
    trace_max_sr = go.Scatter(
        x=[max_sr_vol],
        y=[max_sr_r],
        mode='markers',
        name='Max Sharpe',
        marker=dict(color='red', size=12, symbol='star')
    )
    trace_min_vol = go.Scatter(
        x=[min_vol_vol],
        y=[min_vol_r],
        mode='markers',
        name='Min Volatility',
        marker=dict(color='green', size=12, symbol='star')
    )

    data_traces = [trace_ef, trace_max_sr, trace_min_vol]

    if user_weights is not None and len(user_weights)==len(mean_returns):
        p_r, p_v = portfolio_performance(user_weights, mean_returns, cov_matrix)
        p_r *= 100
        p_v *= 100
        trace_user = go.Scatter(
            x=[p_v],
            y=[p_r],
            mode='markers',
            name='Your Portfolio',
            marker=dict(color='orange', size=14, symbol='diamond')
        )
        data_traces.append(trace_user)

    if bench_return is not None and bench_vol is not None:
        trace_bench = go.Scatter(
            x=[bench_vol*100],
            y=[bench_return*100],
            mode='markers',
            name=bench_name,
            marker=dict(color='blue', size=14, symbol='triangle-up')
        )
        data_traces.append(trace_bench)

    layout = go.Layout(
        title='Efficient Frontier (with Benchmark)',
        xaxis=dict(title='Volatility (%)'),
        yaxis=dict(title='Return (%)'),
        legend=dict(x=0.02, y=0.98)
    )
    fig = go.Figure(data=data_traces, layout=layout)
    return fig

def mcVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a Pandas Data Series.")

def mcCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a Pandas Data Series.")

def run_monte_carlo_simulations(weights, meanReturns, covMatrix, 
                                initial_investment=10000,
                                mc_sims=100, 
                                T=100):
    meanM = np.full(shape=(T, len(weights)), fill_value=0.0)
    for i, mu in enumerate(meanReturns):
        meanM[:, i] = mu  

    portfolio_sims = np.zeros((T, mc_sims))
    L = np.linalg.cholesky(covMatrix)
    
    for m in range(mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        correlated_daily = meanM + Z @ L.T 
        daily_port_ret = np.sum(weights * correlated_daily, axis=1)
        growth_factors = 1 + daily_port_ret
        portfolio_sims[:, m] = initial_investment * np.cumprod(growth_factors)
    
    return portfolio_sims

def build_and_fetch_data():
    """ 
    1) Build the portfolio DataFrame from session_state
    2) Fetch historical prices
    3) Return final_df, daily_returns, etc. 
    """
    weighting_mode = "market_value" if st.session_state.get("weighting_mode","market_value")=="market_value" else "shares"
    portfolio_df = build_portfolio_df(st.session_state["list_stocks"], weighting_mode=weighting_mode)
    portfolio_df = portfolio_df[portfolio_df['WEIGHT']>0].copy()  # exclude zero weights if any

    years_of_data = st.session_state.get("years_of_data", 1)
    tickers_in_portfolio = portfolio_df['TICKER'].unique().tolist()
    hist_prices = get_historical_data(tickers_in_portfolio, years=years_of_data)

    valid_tickers = hist_prices.columns.tolist()
    final_df = portfolio_df[portfolio_df['TICKER'].isin(valid_tickers)].copy()
    final_df['INDUSTRY'] = final_df['TICKER'].apply(fetch_industry)
    final_df["COUNTRY"] = final_df["TICKER"].apply(fetch_country)

    final_df["PURCHASE_PRICE"] = final_df["TICKER"].apply(
        lambda x: st.session_state["list_stocks"][x][1] if x in st.session_state["list_stocks"] else 0
    )
    final_df["COST_BASIS"] = final_df["TICKER"].apply(
        lambda x: (st.session_state["list_stocks"][x][0] * 
                   st.session_state["list_stocks"][x][1]) if x in st.session_state["list_stocks"] else 0
    )

    final_df["GAIN/LOSS"] = final_df["MARKET_VALUE"] - final_df["COST_BASIS"]
    final_df["GAIN/LOSS (%)"] = np.where(
        final_df["COST_BASIS"] != 0,
        (final_df["GAIN/LOSS"] / final_df["COST_BASIS"]) * 100,
        0
    )

    daily_returns = hist_prices[valid_tickers].pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    user_weights = []
    for t in mean_returns.index:
        row = final_df[final_df['TICKER']==t]
        if not row.empty:
            user_weights.append(row.iloc[0]['WEIGHT'])
        else:
            user_weights.append(0)
    user_weights = np.array(user_weights)
    if user_weights.sum()>0:
        user_weights /= user_weights.sum()

    return final_df, hist_prices, daily_returns, mean_returns, cov_matrix, user_weights

def overview_tab(final_df, hist_prices):
    """ Aperçu Général """
    st.header("Aperçu Général")

    # comparer à un indice
    st.subheader("Comparer le Portefeuille à un Indice de Référence ?")
    do_compare_index = st.checkbox("Activer la comparaison à un indice ?")
    benchmark_list = ['SPY','QQQ','DIA','IWM','VTI']
    chosen_index = None
    bench_return, bench_vol = None, None

    years_data = st.session_state.get("years_of_data", 1)

    if do_compare_index:
        chosen_index = st.selectbox("Choisissez un indice :", benchmark_list, index=0)

    cost_basis_total = final_df["COST_BASIS"].sum()
    market_value_total = final_df["MARKET_VALUE"].sum()
    pct_change = 0
    if cost_basis_total != 0:
        pct_change = ((market_value_total - cost_basis_total) / cost_basis_total) * 100

    fig_indicator = make_subplots(
        rows=1, cols=3,
        specs=[ [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}] ]
    )

    fig_indicator.add_trace(
        go.Indicator(
            mode="number",
            value=cost_basis_total,
            title={"text": "Coût d'Achat (Base)"},
            number={"prefix":"$", "valueformat":",.2f"}
        ),
        row=1, col=1
    )

    fig_indicator.add_trace(
        go.Indicator(
            mode="number",
            value=market_value_total,
            title={"text": "Valeur Marchande"},
            number={"prefix":"$", "valueformat":",.2f"}
        ),
        row=1, col=2
    )

    fig_indicator.add_trace(
        go.Indicator(
            mode="number+delta",
            value=pct_change,
            title={"text": "Évolution (%)"},
            delta={
                "reference": 0,
                "relative": False,
                "valueformat": ".2f",
                "increasing":{"color":"green"},
                "decreasing":{"color":"red"}
            },
            number={"suffix":"%", "valueformat":".2f"}
        ),
        row=1, col=3
    )

    fig_indicator.update_layout(
        title="Valeurs Clés du Portefeuille",
        margin=dict(l=50, r=50, t=80, b=30)
    )
    st.plotly_chart(fig_indicator, use_container_width=True)

    st.subheader("Télécharger votre Portefeuille en Excel")
    df_export = final_df[['TICKER', 'SHARES', 'PURCHASE_PRICE', 'CURRENT_PRICE']].copy()
    df_export['NOM'] = df_export['TICKER'].apply(lambda t: real_names.get(t, t))
    df_export = df_export[['TICKER', 'NOM', 'SHARES', 'PURCHASE_PRICE', 'CURRENT_PRICE']]
    df_export.columns = ['Ticker', 'Nom (si dispo)', 'Quantité', 'Prix d\'Achat', 'Prix Actuel']
    df_export['Prix d\'Achat'] = df_export['Prix d\'Achat'].round(2)
    df_export['Prix Actuel']  = df_export['Prix Actuel'].round(2)

    df_export['Cout Total Achat'] = df_export['Quantité'] * df_export['Prix d\'Achat']
    df_export['Valeur Actuelle']  = df_export['Quantité'] * df_export['Prix Actuel']
    df_export['Evolution (%)'] = np.where(
        df_export['Cout Total Achat'] != 0,
        (df_export['Valeur Actuelle'] - df_export['Cout Total Achat']) / df_export['Cout Total Achat'] * 100,
        0
    )

    df_export['Cout Total Achat']  = df_export['Cout Total Achat'].round(2)
    df_export['Valeur Actuelle']   = df_export['Valeur Actuelle'].round(2)
    df_export['Evolution (%)']     = df_export['Evolution (%)'].round(2)

    df_export.loc["TOTAL", "Ticker"]            = "TOTAL"
    df_export.loc["TOTAL", "Nom (si dispo)"]    = ""
    df_export.loc["TOTAL", "Quantité"]          = df_export["Quantité"].sum()
    df_export.loc["TOTAL", "Prix d'Achat"]      = ""
    df_export.loc["TOTAL", "Prix Actuel"]       = ""
    df_export.loc["TOTAL", "Cout Total Achat"]  = df_export["Cout Total Achat"].sum()
    df_export.loc["TOTAL", "Valeur Actuelle"]   = df_export["Valeur Actuelle"].sum()

    if df_export.loc["TOTAL", "Cout Total Achat"] != 0:
        total_pct = (
            (df_export.loc["TOTAL", "Valeur Actuelle"] - df_export.loc["TOTAL", "Cout Total Achat"])
            / df_export.loc["TOTAL", "Cout Total Achat"]
        ) * 100
        df_export.loc["TOTAL", "Evolution (%)"] = round(total_pct, 2)
    else:
        df_export.loc["TOTAL", "Evolution (%)"] = 0

    excel_bytes = generate_excel_download(df_export, file_name="MonPortefeuille.xlsx")

    excel_bytes = generate_excel_download(df_export, file_name="MonPortefeuille.xlsx")

    st.download_button(
        label="Télécharger en Excel",
        data=excel_bytes,
        file_name="MonPortefeuille.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    fig_alloc = px.pie(
        final_df,
        names='TICKER',
        values='WEIGHT',
        title='Allocation du Portefeuille par Ticker'
    )
    fig_alloc.update_traces(
        hovertemplate="%{label}: %{percent}",
        labels=[get_real_name(t) for t in final_df['TICKER']]
    )

    industry_exposure_df = (
        final_df
        .groupby('INDUSTRY', as_index=False)['WEIGHT']
        .sum()
        .rename(columns={'WEIGHT': 'INDUSTRY_WEIGHT'})
    )
    fig_industry = px.pie(
        industry_exposure_df,
        names='INDUSTRY',
        values='INDUSTRY_WEIGHT',
        title='Allocation du Portefeuille par Secteur (Industrie)'
    )
    fig_industry.update_traces(hovertemplate="%{label}: %{percent}")

    geography_df = (
        final_df
        .groupby('COUNTRY', as_index=False)['WEIGHT']
        .sum()
        .rename(columns={'WEIGHT': 'COUNTRY_WEIGHT'})
    )
    fig_geo = px.pie(
        geography_df,
        names='COUNTRY',
        values='COUNTRY_WEIGHT',
        title='Allocation du Portefeuille par Pays'
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig_alloc, use_container_width=True)
    with col2:
        st.plotly_chart(fig_industry, use_container_width=True)
    with col3:
        st.plotly_chart(fig_geo, use_container_width=True)

    st.subheader("Évolution Historique des Prix")
    valid_tickers = final_df['TICKER'].unique().tolist()
    real_name_options = [get_real_name(t) for t in valid_tickers]
    ticker_name_map = dict(zip(valid_tickers, real_name_options))

    with st.form("ticker_selection_form"):
        selected_names = st.multiselect(
            "Sélectionnez les actions à afficher",
            real_name_options,
            default=real_name_options,
        )
        submitted = st.form_submit_button("Mettre à jour l'affichage")

    if submitted and selected_names:
        name_to_ticker = {v: k for k, v in ticker_name_map.items()}
        selected_tickers = [name_to_ticker[n] for n in selected_names if n in name_to_ticker]

        fig_price = go.Figure()
        for tkr in selected_tickers:
            fig_price.add_trace(
                go.Scatter(
                    x=hist_prices.index,
                    y=hist_prices[tkr],
                    mode='lines',
                    name=get_real_name(tkr),
                )
            )
        fig_price.update_layout(
            title='Prix Historiques',
            xaxis_title='Date',
            yaxis_title='Prix'
        )
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("Prix Normalisés (Base 1 au Début)")
        norm_data = hist_prices[selected_tickers].divide(hist_prices[selected_tickers].iloc[0])
        fig_norm = go.Figure()
        for tkr in selected_tickers:
            fig_norm.add_trace(
                go.Scatter(
                    x=norm_data.index,
                    y=norm_data[tkr],
                    mode='lines',
                    name=get_real_name(tkr),
                )
            )
        fig_norm.update_layout(
            title='Évolution Normalisée des Prix',
            xaxis_title='Date',
            yaxis_title='Prix Normalisé'
        )
        st.plotly_chart(fig_norm, use_container_width=True)

    daily_returns = hist_prices[valid_tickers].pct_change().dropna()
    user_weights = final_df['WEIGHT'].to_numpy()
    user_weights = user_weights / user_weights.sum() if user_weights.sum() > 0 else user_weights
    port_daily_ret = daily_returns[valid_tickers].mul(user_weights, axis=1).sum(axis=1)

    avg_daily_ret = port_daily_ret.mean() * 100
    avg_monthly_ret = port_daily_ret.mean() * 21 * 100
    avg_annual_ret = port_daily_ret.mean() * 252 * 100

    if not do_compare_index:
        st.subheader("Rendements du Portefeuille (Sans Benchmark)")
        fig_rends = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )

        fig_rends.add_trace(
            go.Indicator(
                mode="number",
                value=avg_daily_ret,
                number={'suffix': "%"},
                title={"text":"Rendement Journalier"}
            ),
            row=1, col=1
        )
        fig_rends.add_trace(
            go.Indicator(
                mode="number",
                value=avg_monthly_ret,
                number={'suffix': "%"},
                title={"text":"Rendement Mensuel"}
            ),
            row=1, col=2
        )
        fig_rends.add_trace(
            go.Indicator(
                mode="number",
                value=avg_annual_ret,
                number={'suffix': "%"},
                title={"text":f"Rendement Annuel (sur {years_data} an(s))"}
            ),
            row=1, col=3
        )

        fig_rends.update_layout(title="Rendements Moyens du Portefeuille", margin=dict(l=50, r=50))
        st.plotly_chart(fig_rends, use_container_width=True)

    else:

        st.subheader("Rendements (Journalier, Mensuel, Annuel) - Portefeuille vs. Benchmark")
        start_idx = daily_returns.index[0]
        end_idx = daily_returns.index[-1]

        bench_df = yf.download(chosen_index, start=start_idx, end=end_idx, progress=False)['Close']
        bench_dr = bench_df.pct_change().dropna()
        bench_cum = (1 + bench_dr).cumprod() - 1

        port_cum = (1 + port_daily_ret).cumprod() - 1

        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Scatter(
                x=port_cum.index,
                y=port_cum,
                mode='lines',
                name='Portefeuille'
            )
        )
        fig_compare.add_trace(
            go.Scatter(
                x=bench_cum.index,
                y=bench_cum,
                mode='lines',
                name=get_real_name(chosen_index)
            )
        )
        fig_compare.update_layout(
            title=f"Portefeuille vs. {get_real_name(chosen_index)} (Rendement Cumulé)",
            xaxis_title="Date",
            yaxis_title="Rendement Cumulé"
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        port_final = port_cum.iloc[-1] * 100 if len(port_cum) > 0 else 0
        bench_final = bench_cum.iloc[-1] * 100 if len(bench_cum) > 0 else 0

        cc1, cc2 = st.columns(2)
        with cc1:
            st.write(f"**Rendement Final du Portefeuille (sur {years_data} an(s))**")
            fig_pf = go.Figure(
                go.Indicator(
                    mode="number+delta",
                    value=round(port_final, 2),
                    delta={"reference": 0, "position": "bottom"},
                    number={'suffix': "%"},
                    title={"text": "Portefeuille"}
                )
            )
            st.plotly_chart(fig_pf, use_container_width=True)

        with cc2:
            st.write(f"**Rendement Final de {get_real_name(chosen_index)} (sur {years_data} an(s))**")
            fig_ix = go.Figure(
                go.Indicator(
                    mode="number+delta",
                    value=round(bench_final, 2),
                    delta={"reference": 0, "position": "bottom"},
                    number={'suffix': "%"},
                    title={"text": get_real_name(chosen_index)}
                )
            )
            st.plotly_chart(fig_ix, use_container_width=True)

        st.subheader("Rendements (Journalier, Mensuel, Annuel) - Portefeuille vs. Indice")

        port_avg_daily = port_daily_ret.mean() * 100
        port_avg_monthly = port_daily_ret.mean() * 21 * 100
        port_avg_annual = port_daily_ret.mean() * 252 * 100

        idx_avg_daily = bench_dr.mean() * 100
        idx_avg_monthly = bench_dr.mean() * 21 * 100
        idx_avg_annual = bench_dr.mean() * 252 * 100

        fig_rends_comp = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Daily", "Monthly", f"Annual ({years_data} an(s))",
                "Daily", "Monthly", f"Annual ({years_data} an(s))"
            ],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.2
        )

        fig_rends_comp.add_trace(
            go.Indicator(mode="number", value=port_avg_daily, number={'suffix': "%"}),
            row=1, col=1
        )
        fig_rends_comp.add_trace(
            go.Indicator(mode="number", value=port_avg_monthly, number={'suffix': "%"}),
            row=1, col=2
        )
        fig_rends_comp.add_trace(
            go.Indicator(mode="number", value=port_avg_annual, number={'suffix': "%"}),
            row=1, col=3
        )

        # Row 2 -> Benchmark
        fig_rends_comp.add_trace(
            go.Indicator(mode="number", value=idx_avg_daily, number={'suffix': "%"}),
            row=2, col=1
        )
        fig_rends_comp.add_trace(
            go.Indicator(mode="number", value=idx_avg_monthly, number={'suffix': "%"}),
            row=2, col=2
        )
        fig_rends_comp.add_trace(
            go.Indicator(mode="number", value=idx_avg_annual, number={'suffix': "%"}),
            row=2, col=3
        )

        fig_rends_comp.update_layout(
            title="Rendements Moyens (Portefeuille vs. Benchmark)",
            margin=dict(l=50, r=50)
        )
        st.plotly_chart(fig_rends_comp, use_container_width=True)

        with st.expander("Pourquoi ces chiffres (final vs. annuel) diffèrent-ils ?"):
            st.markdown(
                r"""
                    **Le rendement final** sur 3 ans calcule la croissance totale entre le point de départ et 
                    le point d’arrivée:

                    $$
                    \text{Rendement Final} \;=\; \frac{\text{Valeur Finale} - \text{Valeur Initiale}}{\text{Valeur Initiale}} \times 100\%.
                    $$

                    Par exemple, un rendement final de 14% sur 3 ans signifie que le portefeuille s’est 
                    apprécié de 14% au total depuis son point de départ (il y a 3 ans).

                    En revanche, le **rendement annuel** (annualisé sur 3 ans) se base généralement sur 
                    une moyenne du rendement journalier (ou parfois une moyenne géométrique), puis extrapolée
                    à l’année. Dans ce code, on calcule souvent la moyenne arithmétique des rendements journaliers 
                    que l’on multiplie par 252 (environ le nombre de jours de bourse par an) pour obtenir 
                    une estimation “annuelle”:

                    $$
                    \text{Rendement Annuel} \;\approx\; \bigl(\overline{r_{\text{jour}}} \times 252\bigr) \times 100\%.
                    $$

                    Cette méthode (arithmétique) **ne tient pas compte du cumul** jour après jour. 
                    Un autre calcul, plus exact pour l’annualisation, consiste à prendre la croissance 
                    globale et à la “ramener” à une base annuelle, par un calcul géométrique:

                    $$
                    \text{Rendement Annuel Géométrique} \;=\; \Bigl( (1 + \text{Rendement Final})^{1/\text{NbAnnées}} - 1 \Bigr) \times 100\%.
                    $$

                    Ainsi, un portefeuille peut afficher un “**Rendement Final**” de 14% sur 3 ans, 
                    mais un “**Rendement Annuel**” autour de 4.47% (géométriquement). 
                    Ces écarts viennent de la différence entre la **moyenne arithmétique** 
                    et le **rendement composé** réel.

                    **En résumé**: 
                    - Le “Rendement Final (3 ans)” décrit la hausse (ou baisse) globale sur la période.  
                    - Le “Rendement Annuel (3 ans)” tente d’estimer un pourcentage annuel moyen équivalent 
                    à cette hausse totale, mais peut diverger selon la méthode de calcul 
                    (arithmétique vs. géométrique).
                                    """,
                                    unsafe_allow_html=True
                                )

def mpt_tab(final_df, daily_returns, mean_returns, cov_matrix, user_weights):
    """Théorie Moderne du Portefeuille (TMP)."""
    st.header("Théorie Moderne du Portefeuille (TMP)")

    st.write("""
    **Explications**:
    - La *Frontière Efficiente* représente tous les portefeuilles offrant le meilleur rendement 
      possible pour chaque niveau de risque (volatilité).
    - La *VaR (Value at Risk)* à 5% indique la perte maximum potentielle à 5% de probabilité. 
    - La *CVaR (Conditional Value at Risk)* est la perte moyenne au-delà de la VaR (les pires scénarios).
    - Le *Portefeuille Max Sharpe* est le portefeuille sur la Frontière Efficiente 
      qui maximise le ratio de Sharpe (rendement / risque).
    - Le *Portefeuille Volatilité Minimale* est celui qui a la plus faible volatilité.
    """)

    ef_fig = build_efficient_frontier_figure(
        mean_returns,
        cov_matrix,
        user_weights=user_weights
    )
    st.plotly_chart(ef_fig, use_container_width=True)

    max_sr_res = maximize_sharpe(mean_returns, cov_matrix, risk_free=0.0)
    min_vol_res = minimize_volatility(mean_returns, cov_matrix)

    max_sr_w = max_sr_res['x']
    min_vol_w = min_vol_res['x']

    df_alloc_sr = pd.DataFrame({
        "Ticker": mean_returns.index,
        "Weight": max_sr_w
    })
    df_alloc_sr = df_alloc_sr[df_alloc_sr["Weight"]>1e-5].copy()
    df_alloc_sr["Weight (%)"] = df_alloc_sr["Weight"]*100
    df_alloc_sr.sort_values("Weight (%)", ascending=False, inplace=True)

    df_alloc_min_vol = pd.DataFrame({
        "Ticker": mean_returns.index,
        "Weight": min_vol_w
    })
    df_alloc_min_vol = df_alloc_min_vol[df_alloc_min_vol["Weight"]>1e-5].copy()
    df_alloc_min_vol["Weight (%)"] = df_alloc_min_vol["Weight"]*100
    df_alloc_min_vol.sort_values("Weight (%)", ascending=False, inplace=True)

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        st.subheader("Portefeuille Max Sharpe")
        fig_opt_alloc = px.pie(
            df_alloc_sr,
            names="Ticker",
            values="Weight (%)",
            title="Allocation du Portefeuille Max Sharpe",
            hole=0.3
        )
        st.plotly_chart(fig_opt_alloc, use_container_width=True)

        st.markdown("**Composition (pourcentages) :**")
        for i, row in df_alloc_sr.iterrows():
            st.markdown(f"- **{row['Ticker']}** : {row['Weight (%)']:.2f}%")

    with col_opt2:
        st.subheader("Portefeuille Volatilité Minimale")
        fig_min_vol = px.pie(
            df_alloc_min_vol,
            names="Ticker",
            values="Weight (%)",
            title="Allocation du Portefeuille Vol Min",
            hole=0.3
        )
        st.plotly_chart(fig_min_vol, use_container_width=True)

        st.markdown("**Composition (pourcentages) :**")
        for i, row in df_alloc_min_vol.iterrows():
            st.markdown(f"- **{row['Ticker']}** : {row['Weight (%)']:.2f}%")

    st.subheader("Évolution du Ratio de Sharpe (Glissant)")
    log_returns = np.log(daily_returns+1).dropna()

    TRADING_DAYS = st.slider("Fenêtre (jours) pour le calcul glissant du Sharpe", 10, 252, 60, step=10)
    volatility = log_returns.rolling(window=TRADING_DAYS).std() * np.sqrt(252)
    Rf_annual = 0.01
    Rf_daily = Rf_annual / 252
    sharpe_ratio = ((log_returns.rolling(window=TRADING_DAYS).mean() - Rf_daily)*252) / volatility

    fig_sr = go.Figure()
    for column in sharpe_ratio.columns:
        fig_sr.add_trace(
            go.Scatter(
                x=sharpe_ratio.index,
                y=sharpe_ratio[column],
                name=column
            )
        )
    fig_sr.update_layout(title='Sharpe Ratio (glissant)', xaxis_title='Temps', yaxis_title='Sharpe Ratio')
    st.plotly_chart(fig_sr, use_container_width=True)

    p_ret, p_vol = portfolio_performance(user_weights, mean_returns, cov_matrix)
    p_sr = p_ret/p_vol if p_vol>0 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Rendement Annuel (Portefeuille)**")
        fig_ret = go.Figure(go.Indicator(mode='number', value=round(p_ret*100,2), title={"text":"Rendement (%)"}))
        st.plotly_chart(fig_ret, use_container_width=True)
    with c2:
        st.write("**Volatilité Annuelle (Portefeuille)**")
        fig_vol = go.Figure(go.Indicator(mode='number', value=round(p_vol*100,2), title={"text":"Volatilité (%)"}))
        st.plotly_chart(fig_vol, use_container_width=True)
    with c3:
        st.write("**Ratio de Sharpe** (Rf=0)")
        fig_sri = go.Figure(go.Indicator(mode='number', value=round(p_sr,2), title={"text":"Sharpe Ratio"}))
        st.plotly_chart(fig_sri, use_container_width=True)

    st.subheader("Analyse VaR & CVaR via Monte Carlo")
    st.write("""
    Le Monte Carlo simule différents scénarios d'évolution des prix selon la matrice de variance-covariance,
    pour estimer la distribution finale de la valeur du portefeuille.
    """)
    mc_sims = st.number_input("Nombre de simulations (Monte Carlo)", 100, 5000, 100, step=100)
    mc_days = st.number_input("Horizon de simulation (jours)", 30, 365, 100, step=10)
    initial_portfolio_value = 10000

    portfolio_sims = run_monte_carlo_simulations(
        weights=user_weights,
        meanReturns=mean_returns.values,
        covMatrix=cov_matrix.values,
        initial_investment=initial_portfolio_value,
        mc_sims=mc_sims,
        T=mc_days
    )

    fig_mc = go.Figure()
    days_index = list(range(1, mc_days+1))
    for i in range(mc_sims):
        fig_mc.add_trace(
            go.Scatter(
                x=days_index,
                y=portfolio_sims[:, i],
                mode='lines',
                line=dict(width=1),
                showlegend=False
            )
        )
    fig_mc.update_layout(
        title='Simulation Monte Carlo de la Valeur du Portefeuille',
        xaxis_title='Jours',
        yaxis_title='Valeur du Portefeuille (USD)'
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    portResults = pd.Series(portfolio_sims[-1, :])  
    var_level = 5
    VaR = initial_portfolio_value - mcVaR(portResults, alpha=var_level)
    CVaR = initial_portfolio_value - mcCVaR(portResults, alpha=var_level)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"VaR {var_level}%", value=f"${round(VaR,2)}")
    with col2:
        st.metric(label=f"CVaR {var_level}%", value=f"${round(CVaR,2)}")

def portefeuille_optimaux_tab():
    st.header("Portefeuilles Optimaux")

    st.write("## Sélection du périmètre d'actions :")
    zone_choice = st.selectbox(
        "Zone géographique",
        ["USA", "Europe", "Mix (USA + Europe)", "PEA"],  
        index=2
    )

    years_choice = st.slider(
        "Nombre d'années d'historique à considérer", 
        min_value=1, max_value=5, value=1, step=1
    )

    tickers_usa = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA",
        "NVDA", "BRK-B", "META", "JPM", "JNJ", "V",
        "PG", "UNH", "DIS",
    ]
    tickers_eur = [
        "MC.PA", "NESN.SW", "NOVO-B.CO", "ROG.SW", 
        "SAP.DE", "AZN.L", "ULVR.L", "HSBA.L", 
        "BNP.PA", "TTE.PA", "SIE.DE", "ALV.DE",
        "AI.PA", "SHEL.L",
    ]

    selected_tickers = []

    if zone_choice == "USA":
        selected_tickers = tickers_usa

    elif zone_choice == "Europe":
        selected_tickers = tickers_eur

    elif zone_choice == "Mix (USA + Europe)":
        selected_tickers = tickers_usa + tickers_eur

    elif zone_choice == "PEA":

        pea = pd.read_excel("data/liste_pea.xlsx")  
        pea_isin = list(pea["Code ISIN/ISIN Code"])   

        pea_tickers = []
        for isin_code in pea_isin:
            ticker = get_ticker_from_isin(isin_code)
            if ticker != 'n/a' and ticker is not None:
                pea_tickers.append(ticker)
        
        selected_tickers = pea_tickers

    st.write(f"**Tickers inclus**: {', '.join(selected_tickers)}")

    df_all = get_historical_data(selected_tickers, years=years_choice)
    returns = df_all.pct_change(fill_method=None)
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    max_sr_res = maximize_sharpe(meanReturns, covMatrix, risk_free=0.0, bound=(0,1))
    min_vol_res = minimize_volatility(meanReturns, covMatrix, bound=(0,1))

    max_sr_w = max_sr_res['x']
    min_vol_w = min_vol_res['x']

    df_alloc_sr = pd.DataFrame({"Ticker": meanReturns.index, "Weight": max_sr_w})
    df_alloc_sr = df_alloc_sr[df_alloc_sr["Weight"] > 1e-5].copy()
    df_alloc_sr["Weight (%)"] = df_alloc_sr["Weight"] * 100
    df_alloc_sr.sort_values("Weight (%)", ascending=False, inplace=True)

    df_alloc_min_vol = pd.DataFrame({"Ticker": meanReturns.index, "Weight": min_vol_w})
    df_alloc_min_vol = df_alloc_min_vol[df_alloc_min_vol["Weight"] > 1e-5].copy()
    df_alloc_min_vol["Weight (%)"] = df_alloc_min_vol["Weight"] * 100
    df_alloc_min_vol.sort_values("Weight (%)", ascending=False, inplace=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portefeuille Max Sharpe")
        fig_sr = px.pie(df_alloc_sr, names="Ticker", values="Weight (%)", 
                        title="Allocation Max Sharpe", hole=0.3)
        st.plotly_chart(fig_sr, use_container_width=True)

        st.write("**Composition (pourcentages)**:")
        for _, row in df_alloc_sr.iterrows():
            st.write(f"- **{row['Ticker']}**: {row['Weight (%)']:.2f}%")

    with col2:
        st.subheader("Portefeuille Volatilité Minimale")
        fig_min_vol = px.pie(df_alloc_min_vol, names="Ticker", values="Weight (%)",
                             title="Allocation Volatilité Minimale", hole=0.3)
        st.plotly_chart(fig_min_vol, use_container_width=True)

        st.write("**Composition (pourcentages)**:")
        for _, row in df_alloc_min_vol.iterrows():
            st.write(f"- **{row['Ticker']}**: {row['Weight (%)']:.2f}%")

    st.write("---")
    st.subheader("Performances & Risques (VaR, CVaR)")

    def show_portfolio_stats(weights, daily_rets, label):
        valid_cols = [c for c in daily_rets.columns if c in meanReturns.index]
        
        w_dict = dict(zip(meanReturns.index, weights))
        w_valid = [w_dict.get(c, 0) for c in valid_cols]
        w_valid = pd.Series(w_valid, index=valid_cols)
        total_w = w_valid.sum()
        if total_w > 0:
            w_valid /= total_w

        port_ret_series = daily_rets[valid_cols].mul(w_valid, axis=1).sum(axis=1)
        
        daily_mean = port_ret_series.mean() * 100
        monthly_mean = port_ret_series.mean() * 21 * 100
        annual_mean = port_ret_series.mean() * 252 * 100

        daily_vol = port_ret_series.std()
        monthly_vol = daily_vol * (21 ** 0.5)
        annual_vol = daily_vol * (252 ** 0.5)

        alpha = 5
        var_5 = np.percentile(port_ret_series.dropna(), alpha)
        cvar_5 = port_ret_series[port_ret_series <= var_5].mean()

        st.markdown(f"### {label} Portfolio Metrics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rend. Journalier (moy.)", f"{daily_mean:.2f} %")
        c2.metric("Rend. Mensuel (moy.)", f"{monthly_mean:.2f} %")
        c3.metric("Rend. Annuel (moy.)", f"{annual_mean:.2f} %")

        c4, c5, c6 = st.columns(3)
        c4.metric("Volatilité Journ.", f"{daily_vol*100:.2f} %")
        c5.metric("Volatilité Mens.", f"{monthly_vol*100:.2f} %")
        c6.metric("Volatilité Annuelle", f"{annual_vol*100:.2f} %")

        c7, c8 = st.columns(2)
        c7.metric("VaR(5%) [historique]", f"{var_5*100:.2f} %")
        c8.metric("CVaR(5%) [historique]", f"{cvar_5*100:.2f} %")

        st.markdown("""
        > **Note**: VaR et CVaR sont calculés ici sur la distribution historique 
        > des rendements. Une valeur de VaR 5% négative (ex: -2%) signifie qu'il 
        > y a 5% de chances que la perte dépasse 2% sur une journée.
        """)

    show_portfolio_stats(max_sr_w, returns, "Max Sharpe")
    show_portfolio_stats(min_vol_w, returns, "Min Volatilité")


def main():
    st.title("Analyse de Portefeuille & Théorie Moderne du Portefeuille")

    if "list_stocks" not in st.session_state:
        st.session_state["list_stocks"] = default_portfolio.copy()
    if "weighting_mode" not in st.session_state:
        st.session_state["weighting_mode"] = "market_value"
    if "years_of_data" not in st.session_state:
        st.session_state["years_of_data"] = 1
    if "analysis_started" not in st.session_state:
        st.session_state["analysis_started"] = False

    st.write("Avant de lancer l'analyse, vous pouvez ajuster le portefeuille ci-dessous ou garder le portefeuille par défaut.")
    with st.form("parametres_form"):
        temp_rows = []
        for t, (sh, pr) in st.session_state["list_stocks"].items():
            possible_name = real_names.get(t, t)
            temp_rows.append((t, possible_name, sh, pr))

        temp_df = pd.DataFrame(temp_rows, columns=["Ticker", "Nom (si dispo)", "Quantité", "Prix d'Achat"])
        st.dataframe(temp_df, use_container_width=True)

        st.write("### Ajouter ou Mettre à jour un Ticker")
        new_ticker = st.text_input("Nouveau Ticker", value="")
        new_shares = st.number_input("Quantité", min_value=0, step=1, value=0)
        new_price = st.number_input("Prix d'Achat", min_value=0.0, step=1.0, value=0.0)

        st.write("### Supprimer un Ticker existant")
        remove_options = ["(Aucun)"] + list(st.session_state["list_stocks"].keys())
        remove_ticker = st.selectbox("Choisir un ticker à supprimer", remove_options)

        st.write("### Méthode de calcul des poids")
        st.write("""
        - **Valeur de Marché** : le poids est basé sur la valeur actuelle de chaque ligne dans le portefeuille.
        - **Actions Détenues** : le poids est basé sur le nombre d'actions de chaque ligne, 
          sans tenir compte du prix du marché.
        """)
        weighting_option = st.radio(
            "Choisir la méthode de pondération",
            ["Valeur de Marché", "Actions Détenues"],
            index=0
        )

        st.write("### Nombre d'années d'historique à récupérer")
        years_data = st.slider("Années d'historique", 1, 5, 1, step=1)

        valid = st.form_submit_button("Ok - Lancer l'Analyse")

        if valid:
            if new_ticker.strip() != "":
                st.session_state["list_stocks"][new_ticker.strip()] = (new_shares, new_price)

            if remove_ticker != "(Aucun)":
                st.session_state["list_stocks"].pop(remove_ticker, None)

            st.session_state["weighting_mode"] = "market_value" if weighting_option=="Valeur de Marché" else "shares"
            st.session_state["years_of_data"] = years_data
            st.session_state["analysis_started"] = True

    if st.session_state["analysis_started"]:
        final_df, hist_prices, daily_returns, mean_returns, cov_matrix, user_weights = build_and_fetch_data()

        tab1, tab2, tab3 = st.tabs(["Aperçu Général", "Théorie Moderne du Portefeuille", "Portefeuilles Optimaux"])
        with tab1:
            overview_tab(final_df, hist_prices)
        with tab2:
            mpt_tab(final_df, daily_returns, mean_returns, cov_matrix, user_weights)
        with tab3:
            portefeuille_optimaux_tab()

if __name__ == "__main__":
    main()