import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from anthropic import Anthropic

st.set_page_config(page_title="Análisis Bursátil", layout="wide", page_icon="📊")

# Configurar Anthropic (Claude) API
CLAUDE_AVAILABLE = False
try:
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
    if ANTHROPIC_API_KEY:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        CLAUDE_AVAILABLE = True
except Exception as e:
    st.sidebar.warning(f"⚠️ Claude no disponible: {str(e)}")

# Función para traducir con Claude
def translate_to_spanish_v2(text):
    """Traduce texto al español usando Claude"""
    if not CLAUDE_AVAILABLE or not text or len(text) < 10:
        return None
    
    try:
        # Limitar texto a 3000 caracteres
        text_to_translate = text[:3000] if len(text) > 3000 else text
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"Traduce este texto del inglés al español de forma profesional y clara. Solo devuelve la traducción, sin explicaciones:\n\n{text_to_translate}"
            }]
        )
        
        if message.content and len(message.content) > 0:
            return message.content[0].text.strip()
        else:
            return "Error: No se recibió respuesta del modelo"
            
    except Exception as e:
        return f"Error Claude: {str(e)[:150]}"
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
    }
    
    h1 {
        color: #1d1d1f;
        font-weight: 700;
        font-size: 3.5rem !important;
        letter-spacing: -0.015em;
        margin-bottom: 10px !important;
    }
    
    h2 {
        color: #1d1d1f;
        font-weight: 600;
        font-size: 2rem !important;
        letter-spacing: -0.01em;
    }
    
    h3 {
        color: #1d1d1f;
        font-weight: 600;
        font-size: 1.5rem !important;
    }
    
    h4 {
        color: #424245;
        font-weight: 600;
        font-size: 1.2rem !important;
    }
    
    p, .stMarkdown {
        color: #1d1d1f;
        font-weight: 400;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 600;
        color: #1d1d1f;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #6e6e73;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 16px;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 8px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #424245;
        padding: 12px 24px;
        font-weight: 500;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0071e3;
        color: white !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #fbfbfd;
    }
    
    .stTextInput > div > div > input {
        background-color: white;
        border: 1px solid #d2d2d7;
        border-radius: 10px;
        padding: 12px 16px;
        color: #1d1d1f;
        font-size: 16px;
    }
    
    .stSelectbox > div > div > div {
        background-color: white;
        border: 1px solid #d2d2d7;
        border-radius: 10px;
        color: #1d1d1f;
    }
    
    .stButton > button {
        background: #0071e3;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        font-weight: 500;
        font-size: 16px;
    }
    
    .stButton > button:hover {
        background: #0077ed;
    }
    
    .stExpander {
        background-color: white;
        border-radius: 12px;
        border: 1px solid #d2d2d7;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .element-container {
        color: #1d1d1f;
    }
    
    code {
        background-color: #f5f5f7;
        color: #0071e3;
        padding: 4px 8px;
        border-radius: 6px;
        font-family: 'SF Mono', Monaco, monospace;
    }
    
    .translation-box {
        background-color: #f5f5f7;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        border-left: 4px solid #0071e3;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>📊 Análisis Bursátil</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6e6e73; font-size: 1.2rem; margin-bottom: 40px;'>Análisis financiero profesional en tiempo real</p>", unsafe_allow_html=True)

# Sidebar con estilo mejorado
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='font-size: 2.5rem; margin: 0; background: linear-gradient(120deg, #0071e3 0%, #00c4cc 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;'>
            📊
        </h1>
        <h3 style='color: #1d1d1f; margin: 10px 0 5px 0; font-weight: 600;'>Análisis Bursátil</h3>
        <p style='color: #86868b; font-size: 0.85rem; margin: 0;'>Análisis profesional en tiempo real</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Input de ticker con estilo
st.sidebar.markdown("### 🔍 Buscar Acción")
ticker = st.sidebar.text_input(
    "Ingresa el ticker",
    value="AAPL",
    placeholder="Ej: AAPL, MSFT, GOOGL",
    label_visibility="collapsed"
).upper().strip()

# Selector de período
st.sidebar.markdown("### 📅 Período de Análisis")
periodo = st.sidebar.selectbox(
    "Selecciona el período",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Ejemplos populares con botones clickeables
st.sidebar.markdown("### ⭐ Acciones Populares")

# Crear botones en grid
col1, col2, col3 = st.sidebar.columns(3)

popular_stocks = {
    "AAPL": "🍎",
    "MSFT": "🪟",
    "GOOGL": "🔍",
    "TSLA": "🚗",
    "NVDA": "🎮",
    "META": "📱"
}

# CSS para los botones
st.sidebar.markdown("""
    <style>
    .stock-button {
        display: inline-block;
        padding: 8px 12px;
        margin: 4px 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    .stock-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Mostrar botones
button_html = "<div style='text-align: center;'>"
for stock, emoji in popular_stocks.items():
    button_html += f"<span class='stock-button'>{emoji} {stock}</span> "
button_html += "</div>"
st.sidebar.markdown(button_html, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Información adicional
st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 15px; border-radius: 12px; margin-top: 20px;'>
        <p style='margin: 0; font-size: 0.85rem; color: #1d1d1f;'>
            <strong>💡 Consejo:</strong><br>
            Cambia el período para ver diferentes perspectivas del mercado
        </p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Footer del sidebar
st.sidebar.markdown("""
    <div style='text-align: center; padding-top: 20px; border-top: 1px solid #d2d2d7;'>
        <p style='color: #86868b; font-size: 0.75rem; margin: 5px 0;'>
            Powered by<br>
            <strong>Yahoo Finance & Claude AI</strong>
        </p>
    </div>
""", unsafe_allow_html=True)

if not ticker:
    st.warning("⚠️ Por favor ingresa un ticker")
    st.stop()

# Función para traducir con Gemini
@st.cache_data(ttl=86400, show_spinner=False)
def translate_to_spanish(text):
    """Traduce texto al español usando Gemini"""
    if not GEMINI_AVAILABLE or not text:
        return None
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Traduce el siguiente texto al español de manera profesional y precisa. 
        Solo devuelve la traducción, sin explicaciones adicionales:
        
        {text}"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

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

# ============ INFORMACIÓN DE LA EMPRESA ============
st.markdown(f"## 🏢 {info.get('longName', ticker)}")
st.markdown(f"<p style='color: #6e6e73; font-size: 1.1rem;'>{ticker}</p>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**🌍 General**")
    st.markdown(f"<div style='color: #1d1d1f;'><b>Sector:</b> {info.get('sector', 'N/A')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color: #1d1d1f;'><b>Industria:</b> {info.get('industry', 'N/A')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color: #1d1d1f;'><b>País:</b> {info.get('country', 'N/A')}</div>", unsafe_allow_html=True)

with col2:
    st.markdown("**👥 Empresa**")
    employees = info.get('fullTimeEmployees')
    emp_text = f"{employees:,}" if employees else "N/A"
    st.markdown(f"<div style='color: #1d1d1f;'><b>Empleados:</b> {emp_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color: #1d1d1f;'><b>Exchange:</b> {info.get('exchange', 'N/A')}</div>", unsafe_allow_html=True)
    website = info.get('website', 'N/A')
    if website != 'N/A':
        st.markdown(f"<div style='color: #1d1d1f;'><b>Web:</b> <a href='{website}' style='color: #0071e3;'>{website}</a></div>", unsafe_allow_html=True)

with col3:
    st.markdown("**📊 Rango 52 Semanas**")
    high_52 = info.get('fiftyTwoWeekHigh', 0)
    low_52 = info.get('fiftyTwoWeekLow', 0)
    high_text = f"${high_52:.2f}" if high_52 else "N/A"
    low_text = f"${low_52:.2f}" if low_52 else "N/A"
    st.markdown(f"<div style='color: #1d1d1f;'><b>Máximo:</b> {high_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color: #1d1d1f;'><b>Mínimo:</b> {low_text}</div>", unsafe_allow_html=True)
    market_cap = info.get('marketCap')
    cap_text = f"${market_cap/1e9:.2f}B" if market_cap else "N/A"
    st.markdown(f"<div style='color: #1d1d1f;'><b>Cap. Mercado:</b> {cap_text}</div>", unsafe_allow_html=True)

with col4:
    st.markdown("**📈 Métricas Clave**")
    pe = info.get('trailingPE')
    pe_text = f"{pe:.2f}" if pe else "N/A"
    st.markdown(f"<div style='color: #1d1d1f;'><b>P/E Ratio:</b> {pe_text}</div>", unsafe_allow_html=True)
    beta = info.get('beta')
    beta_text = f"{beta:.2f}" if beta else "N/A"
    st.markdown(f"<div style='color: #1d1d1f;'><b>Beta:</b> {beta_text}</div>", unsafe_allow_html=True)
    div_yield = info.get('dividendYield')
    div_text = f"{div_yield*100:.2f}%" if div_yield else "N/A"
    st.markdown(f"<div style='color: #1d1d1f;'><b>Div. Yield:</b> {div_text}</div>", unsafe_allow_html=True)

# Descripción con traducción
summary = info.get('longBusinessSummary')
if summary:
    with st.expander("📄 Ver descripción de la empresa"):
        # Indicador de versión - DEBUGGING
        st.caption("🔄 Powered by Claude AI - Anthropic")
        
        st.markdown("**Descripción Original (Inglés):**")
        st.markdown(f"<div style='color: #1d1d1f;'>{summary}</div>", unsafe_allow_html=True)
        
        # Traducción con Claude
        if CLAUDE_AVAILABLE:
            with st.spinner("🤖 Traduciendo con Claude AI..."):
                traduccion = translate_to_spanish_v2(summary)
                
                if traduccion and not traduccion.startswith("Error"):
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='translation-box'>", unsafe_allow_html=True)
                    st.markdown("**📝 Traducción al Español (powered by Claude):**")
                    st.markdown(f"<div style='color: #1d1d1f;'>{traduccion}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                elif traduccion:
                    st.error(f"❌ {traduccion}")
                    st.info("💡 Verifica tu API key de Anthropic en Settings → Secrets")
                else:
                    st.warning("⚠️ No se pudo traducir")
        else:
            st.info("🔑 **Para activar la traducción con Claude:**\n\n1. Ve a Settings → Secrets\n2. Agrega: `ANTHROPIC_API_KEY = \"tu-api-key\"`")

st.markdown("<br>", unsafe_allow_html=True)

# ============ MÉTRICAS PRINCIPALES ============
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

# Calcular Bandas de Bollinger
hist['BB_middle'] = hist['Close'].rolling(window=20).mean()
bb_std = hist['Close'].rolling(window=20).std()
hist['BB_upper'] = hist['BB_middle'] + (bb_std * 2)
hist['BB_lower'] = hist['BB_middle'] - (bb_std * 2)

# Calcular MACD
exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
hist['MACD'] = exp1 - exp2
hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
hist['MACD_hist'] = hist['MACD'] - hist['Signal']

precio = float(hist['Close'].iloc[-1])
precio_anterior = float(hist['Close'].iloc[-2]) if len(hist) > 1 else precio
cambio = ((precio / precio_anterior) - 1) * 100
maximo = float(hist['High'].max())
minimo = float(hist['Low'].min())
volumen_prom = int(hist['Volume'].mean())
rsi_actual = hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else 0

st.markdown("### Métricas en Tiempo Real")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("💰 Precio Actual", f"${precio:.2f}", f"{cambio:+.2f}%")
col2.metric("📈 Máximo", f"${maximo:.2f}")
col3.metric("📉 Mínimo", f"${minimo:.2f}")
col4.metric("📊 Vol. Promedio", f"{volumen_prom/1e6:.1f}M")
col5.metric("🎯 RSI", f"{rsi_actual:.1f}")

# Cargar S&P 500 para comparaciones
sp500 = get_sp500_data(periodo)

# Calcular variables necesarias para tabs avanzados
if sp500 is not None and not sp500.empty:
    ret_stock = ((float(hist['Close'].iloc[-1]) / float(hist['Close'].iloc[0])) - 1) * 100
    ret_sp500 = ((float(sp500['Close'].iloc[-1]) / float(sp500['Close'].iloc[0])) - 1) * 100
else:
    ret_stock = ((float(hist['Close'].iloc[-1]) / float(hist['Close'].iloc[0])) - 1) * 100
    ret_sp500 = 0

returns_stock = hist['Close'].pct_change().dropna()
vol_anual = returns_stock.std() * np.sqrt(252) * 100

st.markdown("<br>", unsafe_allow_html=True)

# ============ TABS CON HERRAMIENTAS AVANZADAS ============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Análisis Técnico", 
    "📊 Comparativa S&P 500", 
    "💰 Indicadores Financieros",
    "🤖 Análisis AI & Noticias",
    "🛠️ Herramientas Avanzadas"
])

with tab1:
    st.markdown("### Análisis Técnico Avanzado")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{ticker} - Gráfica de Velas con Indicadores',
            'Volumen de Operaciones',
            'MACD (Moving Average Convergence Divergence)',
            'RSI (Relative Strength Index)'
        ),
        row_heights=[0.45, 0.15, 0.2, 0.2]
    )

    # 1. GRÁFICA DE VELAS (CANDLESTICK)
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Precio',
            increasing_line_color='#30d158',
            decreasing_line_color='#ff375f',
            increasing_fillcolor='#30d158',
            decreasing_fillcolor='#ff375f'
        ),
        row=1, col=1
    )
    
    # Bandas de Bollinger
    if not hist['BB_upper'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist['BB_upper'],
                name='BB Superior',
                line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dot'),
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist['BB_middle'],
                name='BB Media (SMA 20)',
                line=dict(color='#0071e3', width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist['BB_lower'],
                name='BB Inferior',
                line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(0, 113, 227, 0.05)',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # SMA 50
    if not hist['SMA_50'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist['SMA_50'],
                name='SMA 50',
                line=dict(color='#ff9500', width=2.5, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )

    # 2. VOLUMEN CON GRADIENTE
    colors = ['rgba(255, 55, 95, 0.7)' if hist['Close'].iloc[i] < hist['Open'].iloc[i] 
              else 'rgba(48, 209, 88, 0.7)' for i in range(len(hist))]

    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volumen',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

    # 3. MACD
    if not hist['MACD'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist['MACD'],
                name='MACD',
                line=dict(color='#0071e3', width=2)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist['Signal'],
                name='Señal',
                line=dict(color='#ff9500', width=2)
            ),
            row=3, col=1
        )
        
        # Histograma MACD
        macd_colors = ['#30d158' if val >= 0 else '#ff375f' for val in hist['MACD_hist']]
        fig.add_trace(
            go.Bar(
                x=hist.index, y=hist['MACD_hist'],
                name='Histograma',
                marker_color=macd_colors,
                showlegend=False
            ),
            row=3, col=1
        )

    # 4. RSI CON ZONAS COLOREADAS
    if not hist['RSI'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=hist['RSI'],
                name='RSI',
                line=dict(color='#bf5af2', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(191, 90, 242, 0.1)'
            ),
            row=4, col=1
        )
        
        # Zonas de sobrecompra y sobreventa
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255, 55, 95, 0.1)", 
                     layer="below", line_width=0, row=4, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(48, 209, 88, 0.1)", 
                     layer="below", line_width=0, row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff375f", 
                     row=4, col=1, opacity=0.6, line_width=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#86868b", 
                     row=4, col=1, opacity=0.4, line_width=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#30d158", 
                     row=4, col=1, opacity=0.6, line_width=1)

    # Configuración profesional del layout
    fig.update_layout(
        height=950,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        paper_bgcolor='white',
        plot_bgcolor='#fafafa',
        font=dict(family="SF Pro Display, -apple-system, sans-serif", size=11, color="#1d1d1f"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        xaxis_rangeslider_visible=False
    )
    
    # Mejorar apariencia de los ejes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.2)',
        showline=True,
        linewidth=1,
        linecolor='#d2d2d7'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.2)',
        showline=True,
        linewidth=1,
        linecolor='#d2d2d7'
    )
    
    # Títulos de ejes
    fig.update_xaxes(title_text="Fecha", row=4, col=1, title_font=dict(size=12, color="#424245"))
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1, title_font=dict(size=11, color="#424245"))
    fig.update_yaxes(title_text="Volumen", row=2, col=1, title_font=dict(size=11, color="#424245"))
    fig.update_yaxes(title_text="MACD", row=3, col=1, title_font=dict(size=11, color="#424245"))
    fig.update_yaxes(title_text="RSI", row=4, col=1, title_font=dict(size=11, color="#424245"))

    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis técnico resumido
    st.markdown("#### 🎯 Señales Técnicas")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rsi_signal = "🟢 COMPRA" if rsi_actual < 30 else "🔴 VENTA" if rsi_actual > 70 else "🟡 NEUTRAL"
        st.metric("RSI", f"{rsi_actual:.1f}", rsi_signal)
    
    with col2:
        if not pd.isna(hist['SMA_50'].iloc[-1]):
            sma_signal = "🟢 ALCISTA" if precio > hist['SMA_50'].iloc[-1] else "🔴 BAJISTA"
            st.metric("SMA 50", f"${hist['SMA_50'].iloc[-1]:.2f}", sma_signal)
        else:
            st.metric("SMA 50", "N/A", "Sin datos")
    
    with col3:
        if not pd.isna(hist['MACD'].iloc[-1]) and not pd.isna(hist['Signal'].iloc[-1]):
            macd_signal = "🟢 POSITIVO" if hist['MACD'].iloc[-1] > hist['Signal'].iloc[-1] else "🔴 NEGATIVO"
            st.metric("MACD", f"{hist['MACD'].iloc[-1]:.2f}", macd_signal)
        else:
            st.metric("MACD", "N/A", "Sin datos")
    
    with col4:
        if not pd.isna(hist['BB_upper'].iloc[-1]) and not pd.isna(hist['BB_lower'].iloc[-1]):
            bb_position = (precio - hist['BB_lower'].iloc[-1]) / (hist['BB_upper'].iloc[-1] - hist['BB_lower'].iloc[-1]) * 100
            bb_signal = "🟢 CERCA INFERIOR" if bb_position < 20 else "🔴 CERCA SUPERIOR" if bb_position > 80 else "🟡 EN MEDIO"
            st.metric("Bollinger", f"{bb_position:.0f}%", bb_signal)
        else:
            st.metric("Bollinger", "N/A", "Sin datos")

with tab2:
    st.markdown("### Comparativa vs S&P 500")

    if sp500 is not None and not sp500.empty:
        norm_stock = (hist['Close'] / float(hist['Close'].iloc[0])) * 100
        norm_sp500 = (sp500['Close'] / float(sp500['Close'].iloc[0])) * 100
        
        # Calcular outperformance
        outperformance = norm_stock - norm_sp500
        
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'Rendimiento Acumulado (Base 100)',
                'Outperformance vs S&P 500'
            ),
            row_heights=[0.65, 0.35]
        )
        
        # Gráfica principal
        fig2.add_trace(go.Scatter(
            x=hist.index, y=norm_stock, name=ticker,
            line=dict(color='#0071e3', width=3.5),
            fill='tozeroy',
            fillcolor='rgba(0, 113, 227, 0.08)',
            hovertemplate='<b>%{x}</b><br>' + ticker + ': %{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        fig2.add_trace(go.Scatter(
            x=sp500.index, y=norm_sp500, name='S&P 500',
            line=dict(color='#ff9500', width=3, dash='dot'),
            hovertemplate='<b>%{x}</b><br>S&P 500: %{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        # Línea de referencia en 100
        fig2.add_hline(y=100, line_dash="dot", line_color="#86868b", 
                      row=1, col=1, opacity=0.5, line_width=1)
        
        # Outperformance
        colors_out = ['rgba(48, 209, 88, 0.6)' if val >= 0 else 'rgba(255, 55, 95, 0.6)' 
                     for val in outperformance]
        
        fig2.add_trace(go.Bar(
            x=hist.index, y=outperformance,
            name='Diferencia',
            marker_color=colors_out,
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Diferencia: %{y:.2f}%<extra></extra>'
        ), row=2, col=1)
        
        fig2.add_hline(y=0, line_dash="solid", line_color="#1d1d1f", 
                      row=2, col=1, opacity=0.8, line_width=1.5)
        
        fig2.update_layout(
            height=600,
            template='plotly_white',
            hovermode='x unified',
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            font=dict(family="SF Pro Display, -apple-system, sans-serif", size=11, color="#1d1d1f"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )
        
        fig2.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)',
            showline=True, linewidth=1, linecolor='#d2d2d7',
            title_text="Fecha", row=2, col=1, title_font=dict(size=12, color="#424245")
        )
        fig2.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)',
            showline=True, linewidth=1, linecolor='#d2d2d7'
        )
        fig2.update_yaxes(title_text="Base 100", row=1, col=1, title_font=dict(size=11, color="#424245"))
        fig2.update_yaxes(title_text="Diferencia (%)", row=2, col=1, title_font=dict(size=11, color="#424245"))
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Métricas de rendimiento mejoradas
        # Calcular correlación y beta
        returns_sp500 = sp500['Close'].pct_change().dropna()
        common_dates = returns_stock.index.intersection(returns_sp500.index)
        if len(common_dates) > 20:
            ret_s = returns_stock.loc[common_dates]
            ret_sp = returns_sp500.loc[common_dates]
            correlation = ret_s.corr(ret_sp)
            covariance = ret_s.cov(ret_sp)
            variance_sp = ret_sp.var()
            beta = covariance / variance_sp if variance_sp != 0 else 0
        else:
            correlation = 0
            beta = 0
        
        # Sharpe Ratio
        risk_free_rate = 4.0 / 100
        sharpe = (ret_stock/100 - risk_free_rate) / (vol_anual/100) if vol_anual > 0 else 0
        
        st.markdown("#### 📊 Análisis Comparativo Detallado")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(f"📈 {ticker}", f"{ret_stock:+.2f}%")
        col2.metric("📊 S&P 500", f"{ret_sp500:+.2f}%")
        col3.metric("⚡ Alpha", f"{ret_stock - ret_sp500:+.2f}%", 
                   delta="Positivo" if ret_stock > ret_sp500 else "Negativo")
        col4.metric("📉 Beta", f"{beta:.2f}")
        col5.metric("🎯 Sharpe", f"{sharpe:.2f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Volatilidad Anual", f"{vol_anual:.2f}%")
        col2.metric("🔗 Correlación", f"{correlation:.2f}")
        
        # Análisis textual
        if ret_stock > ret_sp500:
            performance_text = f"✅ **{ticker} ha superado al S&P 500** por {ret_stock - ret_sp500:.2f} puntos porcentuales."
        else:
            performance_text = f"⚠️ **{ticker} está por debajo del S&P 500** por {ret_sp500 - ret_stock:.2f} puntos porcentuales."
        
        st.info(performance_text)
        
    else:
        st.warning("⚠️ No se pudo cargar S&P 500")

with tab3:
    st.markdown("### Indicadores Financieros")
    
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
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💎 Dividendos")
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

# ============ TAB 4: ANÁLISIS AI & NOTICIAS ============
with tab4:
    st.markdown("### 🤖 Análisis Inteligente con IA")
    
    # Generar análisis AI
    if st.button("🚀 Generar Análisis AI Completo", type="primary", use_container_width=True):
        with st.spinner("🤖 Claude AI está analizando la empresa..."):
            if CLAUDE_AVAILABLE:
                try:
                    # Preparar datos para análisis
                    datos_empresa = f"""
Analiza la siguiente empresa:
- Nombre: {info.get('longName', ticker)}
- Ticker: {ticker}
- Sector: {info.get('sector', 'N/A')}
- Precio actual: ${precio:.2f}
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Market Cap: ${info.get('marketCap', 0)/1e9:.2f}B
- Beta: {info.get('beta', 'N/A')}
- ROE: {info.get('returnOnEquity', 0)*100:.2f}%
- Debt/Equity: {info.get('debtToEquity', 'N/A')}
- Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%
- Rendimiento últimos {periodo}: {ret_stock:.2f}%
- Volatilidad: {vol_anual:.1f}%
- RSI: {rsi_actual:.1f}
"""
                    
                    prompt = f"""{datos_empresa}

Genera un análisis financiero profesional en español con:

1. **Resumen Ejecutivo** (3-4 líneas)
2. **Análisis Fundamental** (fortalezas y debilidades)
3. **Análisis Técnico** (interpretación de indicadores)
4. **Valoración** (¿está cara o barata?)
5. **Riesgos Principales** (3-4 puntos)
6. **Recomendación** (Compra/Mantener/Venta con justificación)
7. **Precio Objetivo** (estimación razonada)

Sé específico, profesional y basado en datos."""
                    
                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=4000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    if message.content and len(message.content) > 0:
                        analisis = message.content[0].text
                        st.markdown("#### 📋 Reporte de Análisis")
                        st.markdown(f"<div style='background: white; padding: 25px; border-radius: 12px; border-left: 4px solid #0071e3;'>{analisis}</div>", unsafe_allow_html=True)
                        
                        # Disclaimer
                        st.warning("⚠️ **Disclaimer:** Este análisis es generado por IA y no constituye asesoría financiera. Consulta con un profesional antes de invertir.")
                    else:
                        st.error("No se pudo generar el análisis")
                        
                except Exception as e:
                    st.error(f"Error al generar análisis: {str(e)}")
            else:
                st.error("🔑 Claude AI no está disponible. Configura tu API key en Settings → Secrets")
    
    st.markdown("---")
    
    # Noticias en tiempo real
    st.markdown("### 📰 Noticias Recientes")
    
    if st.button("🔍 Buscar Noticias Actuales", use_container_width=True):
        with st.spinner(f"Buscando noticias sobre {ticker}..."):
            try:
                # Aquí usarías web_search pero por ahora lo simulo
                st.info("📰 Funcionalidad de búsqueda de noticias en desarrollo. Próximamente disponible con integración de APIs de noticias.")
                
                # Puedes agregar esto si tienes acceso a web_search:
                # from anthropic import Anthropic
                # client = Anthropic()
                # news = client.web_search(query=f"{ticker} stock news latest")
                
            except Exception as e:
                st.error(f"Error al buscar noticias: {str(e)}")
    
    st.markdown("---")
    
    # Score de Valoración
    st.markdown("### 🎯 Score de Valoración")
    
    col1, col2, col3 = st.columns(3)
    
    # Calcular scores simples
    pe = info.get('trailingPE', 0)
    pe_score = 100 if pe and pe < 15 else 50 if pe and pe < 25 else 25 if pe else 0
    
    roe = info.get('returnOnEquity', 0)
    roe_score = 100 if roe and roe > 0.15 else 50 if roe and roe > 0.10 else 25 if roe else 0
    
    debt_eq = info.get('debtToEquity', 0)
    debt_score = 100 if debt_eq and debt_eq < 0.5 else 50 if debt_eq and debt_eq < 1 else 25 if debt_eq else 0
    
    score_total = (pe_score + roe_score + debt_score) / 3
    
    with col1:
        st.metric("📊 Score Valuación", f"{pe_score}/100")
    with col2:
        st.metric("💰 Score Rentabilidad", f"{roe_score}/100")
    with col3:
        st.metric("🏦 Score Solvencia", f"{debt_score}/100")
    
    # Score total con gauge visual
    score_color = "#30d158" if score_total > 70 else "#ff9500" if score_total > 40 else "#ff375f"
    st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 12px; margin-top: 20px;'>
            <h2 style='color: {score_color}; font-size: 3rem; margin: 0;'>{score_total:.0f}/100</h2>
            <p style='color: #86868b; margin: 5px 0;'>Score Total de Inversión</p>
            <div style='background: #f5f5f7; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 10px;'>
                <div style='background: {score_color}; width: {score_total}%; height: 100%;'></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============ TAB 5: HERRAMIENTAS AVANZADAS ============
with tab5:
    st.markdown("### 🛠️ Calculadoras y Herramientas")
    
    tool_option = st.selectbox(
        "Selecciona una herramienta:",
        ["💰 Calculadora de Inversión", "📊 Comparar Múltiples Acciones", "🎯 Análisis de Riesgo/Retorno"]
    )
    
    if tool_option == "💰 Calculadora de Inversión":
        st.markdown("#### Simulador de Inversión")
        
        col1, col2 = st.columns(2)
        
        with col1:
            inversion_inicial = st.number_input("💵 Inversión Inicial (USD)", min_value=100, max_value=1000000, value=10000, step=100)
            num_acciones = int(inversion_inicial / precio)
            st.info(f"Podrías comprar aproximadamente **{num_acciones:,} acciones** al precio actual")
        
        with col2:
            precio_objetivo = st.number_input("🎯 Precio Objetivo (USD)", min_value=1.0, value=float(precio * 1.2), step=1.0)
            ganancia_potencial = ((precio_objetivo - precio) / precio) * 100
        
        st.markdown("---")
        st.markdown("#### 📊 Resultados de la Simulación")
        
        col1, col2, col3, col4 = st.columns(4)
        
        valor_futuro = num_acciones * precio_objetivo
        ganancia_total = valor_futuro - inversion_inicial
        
        col1.metric("💵 Inversión Inicial", f"${inversion_inicial:,.2f}")
        col2.metric("📈 Valor Futuro", f"${valor_futuro:,.2f}")
        col3.metric("💰 Ganancia/Pérdida", f"${ganancia_total:,.2f}", f"{ganancia_potencial:+.2f}%")
        col4.metric("🎯 ROI", f"{ganancia_potencial:.2f}%")
        
        # Gráfica de proyección
        import plotly.graph_objects as go
        
        precios_proyeccion = [precio, precio * 1.05, precio * 1.10, precio * 1.15, precio_objetivo]
        valores_proyeccion = [p * num_acciones for p in precios_proyeccion]
        
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(
            x=["Hoy", "Escenario 1", "Escenario 2", "Escenario 3", "Objetivo"],
            y=valores_proyeccion,
            mode='lines+markers',
            line=dict(color='#0071e3', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(0, 113, 227, 0.1)'
        ))
        
        fig_proj.update_layout(
            title="Proyección de Valor de Inversión",
            yaxis_title="Valor (USD)",
            height=350,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_proj, use_container_width=True)
    
    elif tool_option == "📊 Comparar Múltiples Acciones":
        st.markdown("#### Comparativa de Acciones")
        
        st.info("💡 Ingresa hasta 5 tickers para comparar (separados por comas)")
        
        tickers_comparar = st.text_input("Tickers a comparar:", value=f"{ticker}, MSFT, GOOGL", placeholder="AAPL, MSFT, GOOGL")
        tickers_list = [t.strip().upper() for t in tickers_comparar.split(",") if t.strip()]
        
        if st.button("📊 Comparar", type="primary") and len(tickers_list) > 1:
            with st.spinner("Cargando datos para comparación..."):
                comparacion_data = []
                
                for tick in tickers_list[:5]:  # Máximo 5
                    try:
                        stock_comp = yf.Ticker(tick)
                        hist_comp = stock_comp.history(period=periodo)
                        info_comp = stock_comp.info
                        
                        if not hist_comp.empty:
                            precio_comp = float(hist_comp['Close'].iloc[-1])
                            ret_comp = ((precio_comp / float(hist_comp['Close'].iloc[0])) - 1) * 100
                            
                            comparacion_data.append({
                                'Ticker': tick,
                                'Precio': f"${precio_comp:.2f}",
                                'Rendimiento': f"{ret_comp:+.2f}%",
                                'P/E': f"{info_comp.get('trailingPE', 0):.2f}" if info_comp.get('trailingPE') else "N/A",
                                'Market Cap': f"${info_comp.get('marketCap', 0)/1e9:.2f}B" if info_comp.get('marketCap') else "N/A",
                                'Beta': f"{info_comp.get('beta', 0):.2f}" if info_comp.get('beta') else "N/A"
                            })
                    except:
                        continue
                
                if comparacion_data:
                    df_comp = pd.DataFrame(comparacion_data)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                    
                    # Gráfica comparativa
                    fig_comp = go.Figure()
                    
                    for tick in tickers_list[:5]:
                        try:
                            hist_c = yf.Ticker(tick).history(period=periodo)
                            if not hist_c.empty:
                                norm = (hist_c['Close'] / float(hist_c['Close'].iloc[0])) * 100
                                fig_comp.add_trace(go.Scatter(
                                    x=hist_c.index,
                                    y=norm,
                                    name=tick,
                                    mode='lines',
                                    line=dict(width=2.5)
                                ))
                        except:
                            continue
                    
                    fig_comp.update_layout(
                        title="Comparativa de Rendimiento (Base 100)",
                        yaxis_title="Rendimiento (%)",
                        height=500,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.error("No se pudieron cargar los datos de comparación")
    
    else:  # Análisis Riesgo/Retorno
        st.markdown("#### 📈 Análisis de Riesgo vs Retorno")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📊 Rendimiento Anualizado", f"{ret_stock:.2f}%")
            st.metric("📉 Volatilidad Anual", f"{vol_anual:.1f}%")
            st.metric("🎯 Sharpe Ratio", f"{((ret_stock/100 - 0.04) / (vol_anual/100)):.2f}" if vol_anual > 0 else "N/A")
        
        with col2:
            riesgo_nivel = "🟢 BAJO" if vol_anual < 20 else "🟡 MEDIO" if vol_anual < 35 else "🔴 ALTO"
            st.metric("⚠️ Nivel de Riesgo", riesgo_nivel)
            
            rendimiento_nivel = "🟢 ALTO" if ret_stock > 15 else "🟡 MODERADO" if ret_stock > 0 else "🔴 NEGATIVO"
            st.metric("📈 Nivel de Rendimiento", rendimiento_nivel)
        
        # Matriz de Riesgo-Retorno
        st.markdown("---")
        st.markdown("#### 🎯 Posición en Matriz Riesgo-Retorno")
        
        fig_matriz = go.Figure()
        
        # Cuadrantes
        fig_matriz.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50, fillcolor="rgba(255, 55, 95, 0.1)", line_width=0)
        fig_matriz.add_shape(type="rect", x0=0, y0=0, x1=50, y1=-50, fillcolor="rgba(255, 149, 0, 0.1)", line_width=0)
        fig_matriz.add_shape(type="rect", x0=0, y0=0, x1=-50, y1=50, fillcolor="rgba(255, 204, 0, 0.1)", line_width=0)
        fig_matriz.add_shape(type="rect", x0=0, y0=0, x1=-50, y1=-50, fillcolor="rgba(48, 209, 88, 0.1)", line_width=0)
        
        # Punto de la acción
        fig_matriz.add_trace(go.Scatter(
            x=[ret_stock],
            y=[vol_anual],
            mode='markers+text',
            marker=dict(size=20, color='#0071e3'),
            text=[ticker],
            textposition='top center',
            name=ticker,
            textfont=dict(size=14, color='#1d1d1f')
        ))
        
        fig_matriz.update_layout(
            title="Matriz Riesgo-Retorno",
            xaxis_title="Rendimiento (%)",
            yaxis_title="Volatilidad (%)",
            height=500,
            template='plotly_white',
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='#1d1d1f'),
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='#1d1d1f')
        )
        
        st.plotly_chart(fig_matriz, use_container_width=True)
        
        # Interpretación
        if ret_stock > 0 and vol_anual < 25:
            interpretacion = "✅ **Excelente:** Alto retorno con volatilidad controlada"
        elif ret_stock > 0 and vol_anual >= 25:
            interpretacion = "⚠️ **Cuidado:** Buenos retornos pero alta volatilidad"
        elif ret_stock <= 0 and vol_anual < 25:
            interpretacion = "⚡ **Revisar:** Bajo retorno aunque volatilidad moderada"
        else:
            interpretacion = "❌ **Alto Riesgo:** Retornos negativos y alta volatilidad"
        
        st.info(interpretacion)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align: center; color: #86868b; font-size: 0.9rem;'>📊 Datos: Yahoo Finance | "
    f"🤖 IA: Claude (Anthropic) | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)
