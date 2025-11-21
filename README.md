# 📊 Plataforma Profesional de Análisis Bursátil

Aplicación web avanzada para análisis financiero profesional con IA, desarrollada con Streamlit y potenciada por Google Gemini AI.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tu-app.streamlit.app)

## 🚀 Características Principales

### 📈 **Análisis Técnico Avanzado**
- **Gráficas de velas (Candlestick)** estilo TradingView
- **Indicadores técnicos:**
  - Bandas de Bollinger
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - SMA 20 y SMA 50
- **4 paneles integrados** con volumen y análisis completo
- **Señales técnicas automáticas** (compra/venta/neutral)

### 📊 **Comparativa vs S&P 500**
- Rendimiento normalizado (Base 100)
- Gráfica de outperformance
- **Métricas avanzadas:**
  - Alpha
  - Beta
  - Correlación
  - Sharpe Ratio
  - Volatilidad anualizada

### 💰 **Indicadores Financieros**
- **Valuación:** P/E, P/B, P/S, PEG, Market Cap
- **Rentabilidad:** ROE, ROA, ROIC, Márgenes
- **Crecimiento:** EPS, Revenue Growth, Earnings Growth
- **Solvencia:** Debt/Equity, Current Ratio, Quick Ratio
- **Dividendos:** Yield, Payout Ratio, Dividendo Anual

### 🤖 **Análisis AI con Gemini** (NUEVO)
- **Reportes automáticos generados por IA:**
  - Resumen ejecutivo
  - Análisis fundamental y técnico
  - Evaluación de riesgos
  - Recomendación de inversión
  - Precio objetivo estimado
- **Sistema de scoring inteligente:**
  - Score de Valuación
  - Score de Rentabilidad  
  - Score de Solvencia
  - Score Total (0-100)

### 🛠️ **Herramientas Avanzadas** (NUEVO)

#### 💰 Calculadora de Inversión
- Simulación de inversiones con capital real
- Cálculo automático de número de acciones
- Proyección de ganancias/pérdidas
- ROI y precio objetivo personalizado
- Gráfica de escenarios de inversión

#### 📊 Comparativa Múltiple
- Comparar hasta 5 acciones simultáneamente
- Tabla con métricas clave
- Gráfica de rendimiento comparativo
- Análisis side-by-side en tiempo real

#### 🎯 Análisis Riesgo-Retorno
- Matriz de riesgo-retorno interactiva
- Clasificación automática por cuadrantes
- Sharpe Ratio
- Interpretación inteligente
- Recomendaciones basadas en perfil de riesgo

### 🌐 **Traducción Automática**
- Descripción de empresas traducida al español
- Powered by Google Gemini AI
- Contexto completo en inglés y español

## 🎨 Diseño

- **Estilo Apple/Fintech** - Minimalista y profesional
- **Tema claro optimizado** para análisis
- **Gráficas interactivas** con Plotly
- **Responsive design** - Funciona en móvil, tablet y desktop
- **Sidebar moderno** con acciones populares

## 🚀 Demo en Vivo

[Ver aplicación →](https://tu-app.streamlit.app)

## 📋 Requisitos

- Python 3.8+
- API Key de Google Gemini (gratuita)

## 🔧 Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/stock-analysis-app.git
cd stock-analysis-app

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar API Key (crear .streamlit/secrets.toml)
mkdir .streamlit
echo 'GEMINI_API_KEY = "tu-api-key-aqui"' > .streamlit/secrets.toml

# Ejecutar
streamlit run app.py
```

## 🔑 Obtener API Key de Gemini

1. Ve a [Google AI Studio](https://aistudio.google.com/apikey)
2. Click en "Create API key"
3. Copia tu API key
4. Agrégala en `.streamlit/secrets.toml`

## 🌐 Despliegue en Streamlit Cloud

1. Sube tu código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. En **Settings → Secrets**, agrega:
   ```toml
   GEMINI_API_KEY = "tu-api-key"
   ```
5. ¡Deploy!

## 📦 Dependencias

```txt
streamlit - Framework web
yfinance - Datos financieros de Yahoo Finance
pandas - Manipulación de datos
plotly - Gráficas interactivas
numpy - Cálculos numéricos
google-generativeai - API de Gemini AI
```

## 📊 Datos y APIs

- **Yahoo Finance** - Datos históricos y fundamentales
- **Google Gemini AI** - Análisis y traducción automática
- Datos actualizados cada 30 minutos (caché)

## 🎯 Casos de Uso

- Análisis profesional de acciones
- Comparación de inversiones
- Simulación de portafolios
- Educación financiera
- Reportes automáticos con IA
- Análisis técnico avanzado

## ⚠️ Disclaimer

Esta aplicación es solo para fines educativos e informativos. Los análisis generados por IA no constituyen asesoría financiera. Consulta con un profesional certificado antes de tomar decisiones de inversión.

## 🛣️ Roadmap

- [ ] Portfolio tracking completo
- [ ] Alertas de precio por email/SMS
- [ ] Análisis de sentimiento de noticias
- [ ] Backtesting de estrategias
- [ ] Exportación de reportes en PDF
- [ ] Integración con brokers
- [ ] Watchlist personalizada con almacenamiento
- [ ] Notificaciones push

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el proyecto
2. Crea tu rama (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles

## 👤 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)

## 🙏 Agradecimientos

- [Yahoo Finance](https://finance.yahoo.com/) - Datos financieros
- [Google Gemini](https://ai.google.dev/) - IA y traducción
- [Streamlit](https://streamlit.io/) - Framework web
- [Plotly](https://plotly.com/) - Visualizaciones

---

⭐ Si te resulta útil, ¡dale una estrella al repo!

**Hecho con ❤️ para la comunidad financiera**
