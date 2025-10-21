import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Análisis Bursátil", layout="wide", page_icon="📊")

# CSS personalizado para diseño moderno
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: #0e1117;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #a0aec0;
    }
    .info-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    h1 {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5rem !important;
    }
    h2 {
        color: #60a5fa;
        font-weight: 700;
    }
    h3 {
        color: #34d399;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 10px;
        color: #94a3b8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Header con gradiente
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>📊 Análisis Bursátil Profesional</h1>", unsafe_allow_html=True)

# Sidebar con estilo
st.sidebar.markdown("### ⚙️ Configuración")
ticker = st.sidebar.text_input("🔍 Ticker", "AAPL").upper().strip()
periodo = st.sidebar.selectbox("📅 Período", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Ejemplos populares:**")
st.sidebar.code("AAPL • MSFT • GOOGL\nTSLA • NVDA • META")

if not ticker:
    st.warning("⚠️ Por favor ingresa un ticker")
    st.stop()

@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            return None, None
        info = stock.info
        return hist, info
    except Exception as e:
        return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def get_sp500_data(period):
    try:
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(period=period)
        return hist if not hist.empty else None
    except:
        return None

with st.spinner(f"🚀 Cargando datos de {ticker}..."):
    hist, info = get_stock_data(ticker, periodo)

if hist is None or hist.empty:
    st.error(f"❌ No se encontró el ticker: {ticker}")
    st.stop()

if info is None:
    info = {}

# ============ INFORMACIÓN DE LA EMPRESA ARRIBA ============
st.markdown(f"## 🏢 {info.get('longName', ticker)}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**🌍 Información General**")
    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
    st.write(f"**Industria:** {info.get('industry', 'N/A')}")
    st.write(f"**País:** {info.get('country', 'N/A')}")

with col2:
    st.markdown("**👥 Empresa**")
    employees = info.get('fullTimeEmployees')
    st.write(f"**Empleados:** {employees:,}" if employees else "**Empleados:** N/A")
    st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
    website = info.get('website', 'N/A')
    if website != 'N/A':
        st.write(f"**Web:** [{website}]({website})")
    else:
        st.write("**Web:** N/A")

with col3:
    st.markdown("**📊 Rango 52 Semanas**")
    high_52 = info.get('fiftyTwoWeekHigh', 0)
    low_52 = info.get('fiftyTwoWeekLow', 0)
    if high_52:
        st.write(f"**Máximo:** ${high_52:.2f}")
    else:
        st.write("**Máximo:** N/A")
    if low_52:
        st.write(f"**Mínimo:** ${low_52:.2f}")
    else:
        st.write("**Mínimo:** N/A")
    market_cap = info.get('marketCap')
    if market_cap:
        st.write(f"**Cap. Mercado:** ${market_cap/1e9:.2f}B")
    else:
        st.write("**Cap. Mercado:** N/A")

with col4:
    st.markdown("**📈 Métricas Clave**")
    pe = info.get('trailingPE')
    st.write(f"**P/E Ratio:** {pe:.2f}" if pe else "**P/E:** N/A")
    beta = info.get('beta')
    st.write(f"**Beta:** {beta:.2f}" if beta else "**Beta:** N/A")
    div_yield = info.get('dividendYield')
    st.write(f"**Div. Yield:** {div_yield*100:.2f}%" if div_yield else "**Div. Yield:** N/A")

# Descripción expandible
summary = info.get('longBusinessSummary')
if summary:
    with st.expander("📄 Ver descripción completa de la empresa"):
        st.write(summary)

st.markdown("---")

# ============ MÉTRICAS PRINCIPALES CON COLORES ============
def calculate_sma(data, window):
    if len(data) >= window:
        return data['Close'].rolling(window=window).mean()
    return pd.Series([np.nan] * len(data), index=data.index)

def calculate_rsi(data, period=14):
    if len(data) < period:
        return pd.Series([np.nan] * len(data), index=data.index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

hist['SMA_20'] = calculate_sma(hist, 20)
hist['SMA_50'] = calculate_sma(hist, 50)
hist['RSI'] = calculate_rsi(hist)

precio = float(hist['Close'].iloc[-1])
precio_anterior = float(hist['Close'].iloc[-2]) if len(hist) > 1 else precio
cambio = ((precio / precio_anterior) - 1) * 100
maximo = float(hist['High'].max())
minimo = float(hist['Low'].min())
volumen_prom = int(hist['Volume'].mean())
rsi_actual = hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else 0

st.markdown("### 💎 Métricas en Tiempo Real")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("💰 Precio Actual", f"${precio:.2f}", f"{cambio:+.2f}%")
col2.metric("📈 Máximo", f"${maximo:.2f}")
col3.metric("📉 Mínimo", f"${minimo:.2f}")
col4.metric("📊 Vol. Promedio", f"{volumen_prom/1e6:.1f}M")

# RSI con color
rsi_color = "🟢" if rsi_actual < 30 else "🔴" if rsi_actual > 70 else "🟡"
col5.metric(f"{rsi_color} RSI (14)", f"{rsi_actual:.1f}")

st.markdown("---")

# ============ TABS PARA ORGANIZAR CONTENIDO ============
tab1, tab2, tab3 = st.tabs(["📈 Análisis Técnico", "📊 Comparativa S&P 500", "💰 Indicadores Financieros"])

with tab1:
    st.markdown("### 🎯 Análisis Técnico Avanzado")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Precio y Medias Móviles', 'Volumen', 'RSI'),
        row_heights=[0.5, 0.25, 0.25]
    )

    # Precio
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['Close'], name='Precio',
                   line=dict(color='#10b981', width=2.5)),
        row=1, col=1
    )

    if not hist['SMA_20'].isna().all():
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['SMA_20'], name='SMA 20',
                       line=dict(color='#f59e0b', width=2, dash='dash')),
            row=1, col=1
        )

    if not hist['SMA_50'].isna().all():
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50',
                       line=dict(color='#ef4444', width=2, dash='dot')),
            row=1, col=1
        )

    # Volumen con colores
    colors = ['#ef4444' if hist['Close'].iloc[i] < hist['Close'].iloc[i-1] else '#10b981' 
              for i in range(1, len(hist))]
    colors.insert(0, '#10b981')

    fig.add_trace(
        go.Bar(x=hist.index, y=hist['Volume'], name='Volumen',
               marker_color=colors),
        row=2, col=1
    )

    # RSI
    if not hist['RSI'].isna().all():
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['RSI'], name='RSI',
                       line=dict(color='#8b5cf6', width=2.5)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=3, col=1, opacity=0.7)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=3, col=1, opacity=0.7)

    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1e293b'
    )
    
    fig.update_xaxes(title_text="Fecha", row=3, col=1)
    fig.update_yaxes(title_text="Precio USD", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🆚 Comparativa vs S&P 500")
    
    sp500 = get_sp500_data(periodo)

    if sp500 is not None and not sp500.empty:
        norm_stock = (hist['Close'] / float(hist['Close'].iloc[0])) * 100
        norm_sp500 = (sp500['Close'] / float(sp500['Close'].iloc[0])) * 100
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hist.index, y=norm_stock, name=ticker,
            line=dict(color='#10b981', width=3),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        fig2.add_trace(go.Scatter(
            x=sp500.index, y=norm_sp500, name='S&P 500',
            line=dict(color='#f59e0b', width=3, dash='dot')
        ))
        
        fig2.update_layout(
            height=500,
            template='plotly_dark',
            hovermode='x unified',
            yaxis_title="Rendimiento (Base 100)",
            xaxis_title="Fecha",
            paper_bgcolor='#0e1117',
            plot_bgcolor='#1e293b'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Métricas de rendimiento
        ret_stock = ((float(hist['Close'].iloc[-1]) / float(hist['Close'].iloc[0])) - 1) * 100
        ret_sp500 = ((float(sp500['Close'].iloc[-1]) / float(sp500['Close'].iloc[0])) - 1) * 100
        returns_stock = hist['Close'].pct_change().dropna()
        vol_anual = returns_stock.std() * np.sqrt(252) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Rendimiento", f"{ret_stock:.2f}%")
        col2.metric("📊 S&P 500", f"{ret_sp500:.2f}%")
        col3.metric("⚡ Alpha", f"{ret_stock - ret_sp500:+.2f}%")
        col4.metric("📉 Volatilidad", f"{vol_anual:.1f}%")
    else:
        st.warning("⚠️ No se pudo cargar S&P 500")

with tab3:
    st.markdown("### 💎 Indicadores Financieros Fundamentales")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 💵 Valuación")
        market_cap = info.get('marketCap')
        if market_cap:
            st.metric("Cap. Mercado", f"${market_cap/1e9:.2f}B")
        
        pe = info.get('trailingPE')
        st.metric("P/E (TTM)", f"{pe:.2f}" if pe else "N/A")
        
        forward_pe = info.get('forwardPE')
        st.metric("Forward P/E", f"{forward_pe:.2f}" if forward_pe else "N/A")
        
        pb = info.get('priceToBook')
        st.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")
        
        ps = info.get('priceToSalesTrailing12Months')
        st.metric("P/S Ratio", f"{ps:.2f}" if ps else "N/A")
        
        peg = info.get('pegRatio')
        st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")

    with col2:
        st.markdown("#### 📊 Rentabilidad")
        
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
        st.markdown("#### 📈 Crecimiento")
        
        eps = info.get('trailingEps')
        st.metric("EPS (TTM)", f"${eps:.2f}" if eps else "N/A")
        
        revenue_growth = info.get('revenueGrowth')
        st.metric("↗️ Ingresos", f"{revenue_growth*100:.2f}%" if revenue_growth else "N/A")
        
        earnings_growth = info.get('earningsGrowth')
        st.metric("↗️ Ganancias", f"{earnings_growth*100:.2f}%" if earnings_growth else "N/A")
        
        debt_equity = info.get('debtToEquity')
        st.metric("Deuda/Equity", f"{debt_equity:.2f}" if debt_equity else "N/A")
        
        current_ratio = info.get('currentRatio')
        st.metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "N/A")
        
        quick_ratio = info.get('quickRatio')
        st.metric("Quick Ratio", f"{quick_ratio:.2f}" if quick_ratio else "N/A")
    
    st.markdown("---")
    
    # Dividendos destacados
    st.markdown("#### 💎 Información de Dividendos")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
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

st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: #64748b;'>📊 Datos: Yahoo Finance | "
    f"🕐 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Desarrollado con ❤️ para análisis financiero</p>",
    unsafe_allow_html=True
)
