import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Análisis Bursátil", layout="wide", page_icon="📈")

st.title("📈 Análisis Bursátil")

# Input
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
periodo = st.sidebar.selectbox("Período", ["1mo", "3mo", "6mo", "1y"], index=2)

@st.cache_data(ttl=1800)
def get_data(tick, per):
    try:
        data = yf.download(tick, period=per, progress=False, auto_adjust=True)
        return data if not data.empty else None
    except:
        return None

@st.cache_data(ttl=1800)
def get_sp500(per):
    try:
        data = yf.download("^GSPC", period=per, progress=False, auto_adjust=True)
        return data if not data.empty else None
    except:
        return None

# Cargar datos
hist = get_data(ticker, periodo)

if hist is None or hist.empty:
    st.error(f"❌ No se encontró: {ticker}")
    st.stop()

# Métricas - convertir a float para evitar errores
try:
    precio = float(hist['Close'].iloc[-1])
    precio_inicio = float(hist['Close'].iloc[0])
    cambio = ((precio / precio_inicio) - 1) * 100
    maximo = float(hist['High'].max())
    minimo = float(hist['Low'].min())
except:
    st.error("Error al procesar datos")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio", f"${precio:.2f}")
col2.metric("Cambio", f"{cambio:+.2f}%")
col3.metric("Máximo", f"${maximo:.2f}")
col4.metric("Mínimo", f"${minimo:.2f}")

# Gráfico
st.subheader(f"Precio de {ticker}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name=ticker, 
                         line=dict(color='#00CC96', width=2)))
fig.update_layout(height=400, template='plotly_white', showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Comparativa S&P 500
sp500 = get_sp500(periodo)

if sp500 is not None and not sp500.empty:
    st.subheader("Comparativa vs S&P 500")
    
    norm_stock = (hist['Close'] / float(hist['Close'].iloc[0])) * 100
    norm_sp500 = (sp500['Close'] / float(sp500['Close'].iloc[0])) * 100
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist.index, y=norm_stock, name=ticker, 
                              line=dict(color='#00CC96', width=2)))
    fig2.add_trace(go.Scatter(x=sp500.index, y=norm_sp500, name='S&P 500',
                              line=dict(color='#FF6692', width=2, dash='dot')))
    fig2.update_layout(height=400, template='plotly_white')
    fig2.update_yaxes(title="Base 100")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Métricas comparativas
    ret_stock = ((float(hist['Close'].iloc[-1]) / float(hist['Close'].iloc[0])) - 1) * 100
    ret_sp500 = ((float(sp500['Close'].iloc[-1]) / float(sp500['Close'].iloc[0])) - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento", f"{ret_stock:.2f}%")
    col2.metric("S&P 500", f"{ret_sp500:.2f}%")
    col3.metric("Diferencia", f"{ret_stock - ret_sp500:+.2f}%")

# Info adicional
if st.checkbox("Ver más información"):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Empresa:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            pe = info.get('trailingPE')
            st.write(f"**P/E:** {pe:.2f}" if pe else "**P/E:** N/A")
        with col2:
            mcap = info.get('marketCap')
            st.write(f"**Cap. Mercado:** ${mcap/1e9:.1f}B" if mcap else "**Cap. Mercado:** N/A")
            st.write(f"**País:** {info.get('country', 'N/A')}")
            beta = info.get('beta')
            st.write(f"**Beta:** {beta:.2f}" if beta else "**Beta:** N/A")
    except:
        st.error("No se pudo cargar información adicional")

st.caption("📊 Datos: Yahoo Finance")
