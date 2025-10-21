import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Análisis Bursátil", layout="wide", page_icon="📈")

st.title("📈 Análisis Bursátil Profesional")

# Sidebar
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
periodo = st.sidebar.selectbox("Período", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
st.sidebar.markdown("---")
st.sidebar.caption("💡 Ejemplos: AAPL, MSFT, GOOGL, TSLA, NVDA")

@st.cache_data(ttl=1800)
def get_data(tick, per):
    try:
        stock = yf.Ticker(tick)
        hist = stock.history(period=per)
        info = stock.info
        return hist, info
    except:
        return None, None

# Cargar datos
with st.spinner(f"Cargando {ticker}..."):
    hist, info = get_data(ticker, periodo)

if hist is None or hist.empty:
    st.error(f"❌ No se encontró el ticker: {ticker}")
    st.stop()

# Calcular indicadores técnicos
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

hist['SMA_20'] = calculate_sma(hist, 20)
hist['SMA_50'] = calculate_sma(hist, 50)
hist['RSI'] = calculate_rsi(hist)

# Métricas principales
precio = float(hist['Close'].iloc[-1])
precio_anterior = float(hist['Close'].iloc[-2])
cambio = ((precio / precio_anterior) - 1) * 100
maximo = float(hist['High'].max())
minimo = float(hist['Low'].min())
volumen_prom = int(hist['Volume'].mean())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Precio Actual", f"${precio:.2f}", f"{cambio:+.2f}%")
col2.metric("Máximo", f"${maximo:.2f}")
col3.metric("Mínimo", f"${minimo:.2f}")
col4.metric("Vol. Promedio", f"{volumen_prom/1e6:.1f}M")
col5.metric("RSI (14)", f"{hist['RSI'].iloc[-1]:.1f}")

st.markdown("---")

# Gráfico principal con volumen
st.subheader(f"📊 Análisis Técnico de {ticker}")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('Precio y Medias Móviles', 'Volumen', 'RSI'),
    row_heights=[0.5, 0.25, 0.25]
)

# Precio y medias móviles
fig.add_trace(
    go.Scatter(x=hist.index, y=hist['Close'], name='Precio',
               line=dict(color='#00CC96', width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20',
               line=dict(color='#FFA15A', width=1.5, dash='dash')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50',
               line=dict(color='#EF553B', width=1.5, dash='dot')),
    row=1, col=1
)

# Volumen
fig.add_trace(
    go.Bar(x=hist.index, y=hist['Volume'], name='Volumen',
           marker_color='rgba(99, 110, 250, 0.5)'),
    row=2, col=1
)

# RSI
fig.add_trace(
    go.Scatter(x=hist.index, y=hist['RSI'], name='RSI',
               line=dict(color='#AB63FA', width=2)),
    row=3, col=1
)
# Líneas de sobrecompra y sobreventa
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)

fig.update_layout(height=800, template='plotly_white', showlegend=True, hovermode='x unified')
fig.update_xaxes(title_text="Fecha", row=3, col=1)
fig.update_yaxes(title_text="Precio USD", row=1, col=1)
fig.update_yaxes(title_text="Volumen", row=2, col=1)
fig.update_yaxes(title_text="RSI", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# Comparativa S&P 500
st.markdown("---")
st.subheader("📈 Comparativa vs S&P 500")

@st.cache_data(ttl=1800)
def get_sp500(per):
    try:
        sp500 = yf.Ticker("^GSPC")
        return sp500.history(period=per)
    except:
        return None

sp500 = get_sp500(periodo)

if sp500 is not None and not sp500.empty:
    # Normalizar precios
    norm_stock = (hist['Close'] / float(hist['Close'].iloc[0])) * 100
    norm_sp500 = (sp500['Close'] / float(sp500['Close'].iloc[0])) * 100
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist.index, y=norm_stock, name=ticker,
        line=dict(color='#00CC96', width=2.5)
    ))
    fig2.add_trace(go.Scatter(
        x=sp500.index, y=norm_sp500, name='S&P 500',
        line=dict(color='#FF6692', width=2, dash='dot')
    ))
    
    fig2.update_layout(
        height=400,
        template='plotly_white',
        hovermode='x unified',
        yaxis_title="Rendimiento (Base 100)",
        xaxis_title="Fecha"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Métricas de rendimiento
    ret_stock = ((float(hist['Close'].iloc[-1]) / float(hist['Close'].iloc[0])) - 1) * 100
    ret_sp500 = ((float(sp500['Close'].iloc[-1]) / float(sp500['Close'].iloc[0])) - 1) * 100
    
    # Volatilidad
    returns_stock = hist['Close'].pct_change().dropna()
    vol_anual = returns_stock.std() * np.sqrt(252) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendimiento", f"{ret_stock:.2f}%")
    col2.metric("S&P 500", f"{ret_sp500:.2f}%")
    col3.metric("Alpha", f"{ret_stock - ret_sp500:+.2f}%")
    col4.metric("Volatilidad Anual", f"{vol_anual:.1f}%")

# Indicadores Financieros
st.markdown("---")
st.subheader("💰 Indicadores Financieros y Fundamentales")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 📊 Valuación")
    market_cap = info.get('marketCap')
    st.metric("Capitalización", f"${market_cap/1e9:.2f}B" if market_cap else "N/A")
    
    pe = info.get('trailingPE')
    st.metric("P/E Ratio (TTM)", f"{pe:.2f}" if pe else "N/A")
    
    forward_pe = info.get('forwardPE')
    st.metric("Forward P/E", f"{forward_pe:.2f}" if forward_pe else "N/A")
    
    pb = info.get('priceToBook')
    st.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")
    
    ps = info.get('priceToSalesTrailing12Months')
    st.metric("P/S Ratio", f"{ps:.2f}" if ps else "N/A")
    
    peg = info.get('pegRatio')
    st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")

with col2:
    st.markdown("##### 💵 Rentabilidad")
    
    roe = info.get('returnOnEquity')
    st.metric("ROE", f"{roe*100:.2f}%" if roe else "N/A")
    
    roa = info.get('returnOnAssets')
    st.metric("ROA", f"{roa*100:.2f}%" if roa else "N/A")
    
    margen_bruto = info.get('grossMargins')
    st.metric("Margen Bruto", f"{margen_bruto*100:.2f}%" if margen_bruto else "N/A")
    
    margen_op = info.get('operatingMargins')
    st.metric("Margen Operativo", f"{margen_op*100:.2f}%" if margen_op else "N/A")
    
    margen_neto = info.get('profitMargins')
    st.metric("Margen Neto", f"{margen_neto*100:.2f}%" if margen_neto else "N/A")
    
    roic = info.get('returnOnCapital')
    st.metric("ROIC", f"{roic*100:.2f}%" if roic else "N/A")

with col3:
    st.markdown("##### 📈 Crecimiento y Deuda")
    
    eps = info.get('trailingEps')
    st.metric("EPS (TTM)", f"${eps:.2f}" if eps else "N/A")
    
    revenue_growth = info.get('revenueGrowth')
    st.metric("Crecimiento Ingresos", f"{revenue_growth*100:.2f}%" if revenue_growth else "N/A")
    
    earnings_growth = info.get('earningsGrowth')
    st.metric("Crecimiento Ganancias", f"{earnings_growth*100:.2f}%" if earnings_growth else "N/A")
    
    debt_equity = info.get('debtToEquity')
    st.metric("Deuda/Equity", f"{debt_equity:.2f}" if debt_equity else "N/A")
    
    current_ratio = info.get('currentRatio')
    st.metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "N/A")
    
    quick_ratio = info.get('quickRatio')
    st.metric("Quick Ratio", f"{quick_ratio:.2f}" if quick_ratio else "N/A")

# Dividendos
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("##### 💎 Dividendos")
    div_yield = info.get('dividendYield')
    st.metric("Dividend Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")

with col2:
    div_rate = info.get('dividendRate')
    st.metric("Dividendo Anual", f"${div_rate:.2f}" if div_rate else "N/A")

with col3:
    payout = info.get('payoutRatio')
    st.metric("Payout Ratio", f"{payout*100:.2f}%" if payout else "N/A")

with col4:
    beta = info.get('beta')
    st.metric("Beta (5Y)", f"{beta:.2f}" if beta else "N/A")

# Información de la empresa
with st.expander("🏢 Información de la Empresa"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Nombre:** {info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industria:** {info.get('industry', 'N/A')}")
        st.write(f"**País:** {info.get('country', 'N/A')}")
        st.write(f"**Ciudad:** {info.get('city', 'N/A')}")
    
    with col2:
        employees = info.get('fullTimeEmployees')
        st.write(f"**Empleados:** {employees:,}" if employees else "**Empleados:** N/A")
        st.write(f"**Website:** {info.get('website', 'N/A')}")
        st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
        st.write(f"**52W High:** ${info.get('fiftyTwoWeekHigh', 0):.2f}")
        st.write(f"**52W Low:** ${info.get('fiftyTwoWeekLow', 0):.2f}")
    
    summary = info.get('longBusinessSummary')
    if summary:
        st.markdown("**Descripción del Negocio:**")
        st.write(summary)

st.markdown("---")
st.caption(f"📊 Datos: Yahoo Finance | 🕐 Última actualización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
