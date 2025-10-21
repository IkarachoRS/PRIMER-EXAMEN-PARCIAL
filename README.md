# 📈 Análisis Bursátil Comparativo

Aplicación web interactiva desarrollada con Streamlit para realizar análisis financiero profesional de acciones con comparativa contra el S&P 500.

## 🎯 Características

- **Búsqueda Inteligente**: Busca empresas por ticker o nombre
- **Información Empresarial**: Resumen completo de la empresa incluyendo sector, industria, empleados y descripción del negocio
- **Análisis Técnico**: Gráficas interactivas de precios históricos con volumen
- **Comparativa S&P 500**: Rendimiento normalizado vs el índice de referencia
- **Métricas de Riesgo**:
  - Volatilidad anualizada
  - Sharpe Ratio
  - Beta
  - Correlación con el mercado
  - Máximo Drawdown
- **Indicadores Financieros**:
  - Ratios de valuación (P/E, P/B, PEG)
  - Métricas de rentabilidad (ROE, ROA, márgenes)
  - Análisis de dividendos y estructura de capital
- **Caché de Datos**: Optimización del rendimiento con actualización cada hora
- **Interfaz Responsive**: Diseño adaptable para diferentes dispositivos

## 🚀 Demo en Vivo

[Ver aplicación en vivo](https://tu-app.streamlit.app) *(Actualizar con tu URL de Streamlit Cloud)*

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 🔧 Instalación Local

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/stock-analysis-app.git
cd stock-analysis-app
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En macOS/Linux
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📦 Dependencias

```
streamlit==1.29.0
yfinance==0.2.32
pandas==2.1.3
plotly==5.18.0
numpy==1.26.2
```

## 💻 Uso

1. **Ingresar Ticker**: En la barra lateral, ingresa el ticker de la empresa (ej: AAPL, TSLA, GOOGL) o su nombre
2. **Seleccionar Período**: Elige el período de análisis (1 mes a histórico completo)
3. **Analizar**: La aplicación mostrará automáticamente:
   - Métricas principales (precio, cambio %, capitalización)
   - Resumen de la empresa
   - Gráfica comparativa vs S&P 500
   - Métricas de rendimiento y riesgo
   - Indicadores financieros detallados

## 📊 Ejemplos de Tickers

| Empresa | Ticker |
|---------|--------|
| Apple | AAPL |
| Microsoft | MSFT |
| Tesla | TSLA |
| Amazon | AMZN |
| Google | GOOGL |
| Meta | META |
| NVIDIA | NVDA |
| Netflix | NFLX |

## 🌐 Despliegue en Streamlit Cloud

1. **Subir a GitHub**
   - Asegúrate de tener todos los archivos (`app.py`, `requirements.txt`, `README.md`)
   - Haz push a tu repositorio

2. **Conectar con Streamlit Cloud**
   - Visita [share.streamlit.io](https://share.streamlit.io)
   - Inicia sesión con GitHub
   - Selecciona tu repositorio
   - Configura:
     - **Main file path**: `app.py`
     - **Python version**: 3.11
   - Click en "Deploy"

3. **Compartir**
   - Tu app estará disponible en: `https://[tu-app-name].streamlit.app`

## 🏗️ Estructura del Proyecto

```
stock-analysis-app/
│
├── app.py                 # Aplicación principal de Streamlit
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Documentación
└── .gitignore            # Archivos a ignorar en Git
```

## 🛠️ Tecnologías Utilizadas

- **[Streamlit](https://streamlit.io/)**: Framework para aplicaciones web de datos
- **[yfinance](https://github.com/ranaroussi/yfinance)**: API para obtener datos financieros de Yahoo Finance
- **[Plotly](https://plotly.com/)**: Visualizaciones interactivas
- **[Pandas](https://pandas.pydata.org/)**: Manipulación y análisis de datos
- **[NumPy](https://numpy.org/)**: Cálculos numéricos

## 📈 Métricas Calculadas

### Rendimiento
- **Rendimiento Total**: Ganancia/pérdida porcentual en el período
- **Rendimiento vs S&P 500**: Diferencial de rendimiento contra el benchmark

### Riesgo
- **Volatilidad**: Desviación estándar anualizada de los retornos
- **Beta**: Sensibilidad de la acción respecto al mercado (S&P 500)
- **Correlación**: Relación lineal con el S&P 500 (-1 a 1)
- **Maximum Drawdown**: Mayor caída desde un máximo histórico

### Eficiencia
- **Sharpe Ratio**: Rendimiento ajustado por riesgo (usando tasa libre de riesgo del 4%)

## ⚠️ Limitaciones

- Los datos provienen de Yahoo Finance y pueden tener un retraso de ~15 minutos
- Algunos tickers internacionales pueden requerir sufijos específicos (ej: `.MX` para México)
- La aplicación no constituye asesoría financiera

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Ideas para Mejoras Futuras

- [ ] Agregar análisis técnico (RSI, MACD, Bandas de Bollinger)
- [ ] Implementar gráficas de velas japonesas
- [ ] Comparativa con múltiples empresas simultáneamente
- [ ] Exportación de reportes en PDF
- [ ] Alertas de precio personalizadas
- [ ] Integración con portfolio tracking
- [ ] Análisis de sentimiento de noticias

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)

## 🙏 Agradecimientos

- [Yahoo Finance](https://finance.yahoo.com/) por proporcionar datos financieros gratuitos
- [Streamlit](https://streamlit.io/) por el increíble framework
- La comunidad de código abierto por las librerías utilizadas

---

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!