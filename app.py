import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Análisis Bursátil", layout="wide", page_icon="📈")

# Título principal
st.title("📈 Análisis Bursátil Comparativo")
st.markdown("---")

# Sidebar para inputs
with st.sidebar:
    st.header("⚙️ Configuración")
    ticker_input = st.text_input("Ticker o Nombre de Empresa", value="AAPL").upper()
    
    periodo = st.selectbox(
        "Período de análisis",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    
    st.markdown("---")
    st.markdown("**Desarrollado para análisis financiero profesional**")

# Función para obtener datos
@st.cache_data(ttl=3600)
def obtener_datos(ticker, periodo):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=periodo)
        info = stock.info
        sp500 = yf.Ticker("^GSPC").history(period=periodo)
        return stock, hist, info, sp500
    except Exception as e:
        return None, None, None, None

# Función para calcular métricas
def calcular_metricas(hist, sp500_hist):
    if hist.empty or sp500_hist.empty:
        return None
    
    # Alinear fechas
    df = pd.DataFrame({
        'stock': hist['Close'],
        'sp500': sp500_hist['Close']
    }).dropna()
    
    if df.empty:
        return None
    
    # Rendimientos
    ret_stock = (df['stock'].iloc[-1] / df['stock'].iloc[0] - 1) * 100
    ret_sp500 = (df['sp500'].iloc[-1] / df['sp500'].iloc[0] - 1) * 100
    
    # Volatilidad anualizada
    daily_returns_stock = df['stock'].pct_change().dropna()
    daily_returns_sp500 = df['sp500'].pct_change().dropna()
    
    vol_stock = daily_returns_stock.std() * np.sqrt(252) * 100
    vol_sp500 = daily_returns_sp500.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (asumiendo tasa libre de riesgo de 4%)
    rf_rate = 4.0
    sharpe_stock = (ret_stock - rf_rate) / vol_stock if vol_stock > 0 else 0
    sharpe_sp500 = (ret_sp500 - rf_rate) / vol_sp500 if vol_sp500 > 0 else 0
    
    # Correlación
    corr = daily_returns_stock.corr(daily_returns_sp500)
    
    # Beta
    covariance = daily_returns_stock.cov(daily_returns_sp500)
    variance_sp500 = daily_returns_sp500.var()
    beta = covariance / variance_sp500 if variance_sp500 > 0 else 0
    
    # Máximo drawdown
    cumulative = (1 + daily_returns_stock).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = ((cumulative - running_max) / running_max * 100).min()
    
    return {
        'ret_stock': ret_stock,
        'ret_sp500': ret_sp500,
        'vol_stock': vol_stock,
        'vol_sp500': vol_sp500,
        'sharpe_stock': sharpe_stock,
        'sharpe_sp500': sharpe_sp500,
        'correlation': corr,
        'beta': beta,
        'max_drawdown': drawdown
    }

# Obtener datos
with st.spinner(f"Obteniendo datos de {ticker_input}..."):
    stock, hist, info, sp500_hist = obtener_datos(ticker_input, periodo)

if stock is None or hist.empty:
    st.error(f"❌ No se encontraron datos para '{ticker_input}'. Verifica el ticker e intenta nuevamente.")
    st.stop()

# Información de la empresa
st.header(f"{info.get('longName', ticker_input)} ({ticker_input})")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Precio Actual", f"${info.get('currentPrice', 'N/A'):,.2f}" if info.get('currentPrice') else "N/A")
with col2:
    change_pct = info.get('regularMarketChangePercent', 0)
    st.metric("Cambio %", f"{change_pct:.2f}%", delta=f"{change_pct:.2f}%")
with col3:
    st.metric("Cap. de Mercado", f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A")
with col4:
    st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A")
with col5:
    st.metric("Volumen", f"{info.get('volume', 0):,.0f}" if info.get('volume') else "N/A")

# Resumen de la empresa
with st.expander("📋 Resumen de la Empresa", expanded=True):
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Industria:** {info.get('industry', 'N/A')}")
        st.markdown(f"**País:** {info.get('country', 'N/A')}")
        st.markdown(f"**Website:** {info.get('website', 'N/A')}")
    
    with col_right:
        st.markdown(f"**Empleados:** {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "**Empleados:** N/A")
        st.markdown(f"**Beta:** {info.get('beta', 'N/A'):.2f}" if info.get('beta') else "**Beta:** N/A")
        st.markdown(f"**52W High:** ${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else "**52W High:** N/A")
        st.markdown(f"**52W Low:** ${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else "**52W Low:** N/A")
    
    if info.get('longBusinessSummary'):
        st.markdown("**Descripción:**")
        st.write(info.get('longBusinessSummary'))

st.markdown("---")

# Gráfico de precios comparativo
st.subheader("📊 Análisis de Precios vs S&P 500")

# Normalizar precios para comparación
hist_norm = (hist['Close'] / hist['Close'].iloc[0] * 100)
sp500_norm = (sp500_hist['Close'] / sp500_hist['Close'].iloc[0] * 100)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('Rendimiento Normalizado (Base 100)', 'Volumen de Operaciones'),
    row_heights=[0.7, 0.3]
)

# Gráfico de precios normalizados
fig.add_trace(
    go.Scatter(x=hist.index, y=hist_norm, name=ticker_input, 
               line=dict(color='#1f77b4', width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=sp500_hist.index, y=sp500_norm, name='S&P 500',
               line=dict(color='#ff7f0e', width=2, dash='dash')),
    row=1, col=1
)

# Gráfico de volumen
fig.add_trace(
    go.Bar(x=hist.index, y=hist['Volume'], name='Volumen',
           marker_color='rgba(31, 119, 180, 0.5)'),
    row=2, col=1
)

fig.update_layout(
    height=700,
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.update_xaxes(title_text="Fecha", row=2, col=1)
fig.update_yaxes(title_text="Índice Base 100", row=1, col=1)
fig.update_yaxes(title_text="Volumen", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# Métricas de rendimiento
st.markdown("---")
st.subheader("📈 Métricas de Rendimiento y Riesgo")

metricas = calcular_metricas(hist, sp500_hist)

if metricas:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"Rendimiento {ticker_input}", 
            f"{metricas['ret_stock']:.2f}%",
            delta=f"{metricas['ret_stock'] - metricas['ret_sp500']:.2f}% vs S&P500"
        )
        st.metric("Volatilidad", f"{metricas['vol_stock']:.2f}%")
    
    with col2:
        st.metric("Rendimiento S&P 500", f"{metricas['ret_sp500']:.2f}%")
        st.metric("Volatilidad S&P 500", f"{metricas['vol_sp500']:.2f}%")
    
    with col3:
        st.metric("Sharpe Ratio", f"{metricas['sharpe_stock']:.2f}")
        st.metric("Beta", f"{metricas['beta']:.2f}")
    
    with col4:
        st.metric("Correlación", f"{metricas['correlation']:.2f}")
        st.metric("Max Drawdown", f"{metricas['max_drawdown']:.2f}%")

# Indicadores financieros
st.markdown("---")
st.subheader("💰 Indicadores Financieros Clave")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### Valuación")
    st.markdown(f"**P/E (TTM):** {info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "**P/E (TTM):** N/A")
    st.markdown(f"**Forward P/E:** {info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else "**Forward P/E:** N/A")
    st.markdown(f"**PEG Ratio:** {info.get('pegRatio', 'N/A'):.2f}" if info.get('pegRatio') else "**PEG Ratio:** N/A")
    st.markdown(f"**P/B Ratio:** {info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else "**P/B Ratio:** N/A")

with col2:
    st.markdown("##### Rentabilidad")
    st.markdown(f"**ROE:** {info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "**ROE:** N/A")
    st.markdown(f"**ROA:** {info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else "**ROA:** N/A")
    st.markdown(f"**Margen Bruto:** {info.get('grossMargins', 0)*100:.2f}%" if info.get('grossMargins') else "**Margen Bruto:** N/A")
    st.markdown(f"**Margen Operativo:** {info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else "**Margen Operativo:** N/A")

with col3:
    st.markdown("##### Dividendos & Deuda")
    st.markdown(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "**Dividend Yield:** N/A")
    st.markdown(f"**Payout Ratio:** {info.get('payoutRatio', 0)*100:.2f}%" if info.get('payoutRatio') else "**Payout Ratio:** N/A")
    st.markdown(f"**Debt/Equity:** {info.get('debtToEquity', 'N/A'):.2f}" if info.get('debtToEquity') else "**Debt/Equity:** N/A")
    st.markdown(f"**Current Ratio:** {info.get('currentRatio', 'N/A'):.2f}" if info.get('currentRatio') else "**Current Ratio:** N/A")

st.markdown("---")
st.caption("📊 Datos proporcionados por Yahoo Finance | Actualización en tiempo real")