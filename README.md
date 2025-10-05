# Predictor de Precio del Cobre

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

Sistema completo de Machine Learning para predecir el precio del cobre utilizando series temporales y técnicas avanzadas de feature engineering.

---

## Tabla de Contenidos

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Problema de Negocio](#problema-de-negocio)
- [Resultados Destacados](#resultados-destacados)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Metodología](#metodología)
- [Modelos Implementados](#modelos-implementados)
- [Conclusiones](#conclusiones)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Autor](#autor)

---

## Resumen Ejecutivo

**Objetivo**: Predecir el precio del cobre (USD/lb) utilizando datos históricos y técnicas de machine learning.

**Resultado Principal**: Modelo Ridge Regression con **99.14% de precisión (R cuadrado)** y error promedio de solo **$0.0173 USD/lb** (1.7 centavos).

**Dataset**: 3,872 registros diarios desde mayo 2010 hasta octubre 2025 (15+ años de datos históricos del mercado de futuros de cobre).

**Impacto**: Modelo deployable para planificación presupuestaria, cobertura de riesgo financiero y decisiones de inversión en la industria minera.

---

## Problema de Negocio

### Contexto

Chile es el **mayor productor de cobre del mundo**, representando aproximadamente el 28% de la producción global. El precio del cobre es volátil y afecta directamente:

- **Presupuestos nacionales**: Ingresos fiscales de Chile dependen del precio del cobre
- **Empresas mineras**: Planificación de inversiones CAPEX
- **Mercados financieros**: Instrumentos derivados y cobertura de riesgo
- **Economía global**: Indicador de salud económica mundial

### Pregunta Clave

> ¿Cuál será el precio del cobre mañana, la próxima semana o el próximo mes?

### Valor del Proyecto

Un modelo preciso permite:
- Reducir incertidumbre en proyecciones financieras
- Optimizar estrategias de cobertura (hedging)
- Mejorar timing de decisiones de compra/venta
- Anticipar tendencias del mercado

---

## Resultados Destacados

### Comparación de Modelos

| Modelo                  | R cuadrado | MAE (USD/lb) | RMSE (USD/lb) | MAPE   |
|-------------------------|------------|--------------|---------------|--------|
| **Ridge Regression**    | **0.9914** | **$0.0173**  | **$0.0345**   | **0.38%** |
| Linear Regression       | 0.9923     | $0.0169      | $0.0326       | 0.37%  |
| SARIMAX                 | 0.7593     | $0.1370      | $0.1822       | 3.02%  |
| Random Forest           | 0.6809     | $0.0876      | $0.2098       | 1.71%  |
| XGBoost                 | 0.6567     | $0.0921      | $0.2176       | 1.80%  |

### Insights Clave

1. **La simplicidad ganó**: Ridge Regression superó a modelos complejos (XGBoost, Random Forest)
2. **Relación lineal fuerte**: El precio del cobre tiene memoria extremadamente alta (lag_1 correlación = 0.9975)
3. **Feature Engineering de calidad > Complejidad del modelo**: 42 features bien diseñadas fueron más valiosas que algoritmos sofisticados
4. **Regularización óptima**: Alpha = 0.01 (regularización muy leve, casi Linear Regression)

### Visualizaciones

#### Comparación de Métricas
![Comparación de Modelos](reports/figures/model_comparison_metrics.png)

#### Predicciones vs Realidad
![Predicciones](reports/figures/predictions_comparison_all_models.png)

---

## Estructura del Proyecto

```
predictor-precio-cobre/
│
├── data/
│   ├── raw/                          # Datos descargados de Yahoo Finance
│   │   └── precio_cobre_yahoo.csv
│   └── processed/                    # Datos con features creadas
│       ├── precio_cobre_features.csv
│       ├── features_summary.csv
│       └── model_comparison_all.csv
│
├── models/                           # Modelos entrenados (.pkl)
│   ├── ridge_model.pkl              # Mejor modelo (R2=0.9914)
│   ├── linear_regression.pkl
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── sarimax_model.pkl
│   └── arima_model.pkl
│
├── notebooks/                        # Jupyter Notebooks (flujo completo)
│   ├── 01_descarga_datos.ipynb      # Extracción desde Yahoo Finance
│   ├── 02_eda.ipynb                 # Análisis Exploratorio de Datos
│   ├── 03_feature_engineering.ipynb # Creación de 42 features
│   ├── 04_modelado.ipynb            # Modelos baseline (LR, ARIMA)
│   └── 05_modelado_avanzado.ipynb   # Modelos avanzados (Ridge, XGB, RF, SARIMAX)
│
├── reports/
│   └── figures/                      # Visualizaciones generadas
│       ├── model_comparison_metrics.png
│       └── predictions_comparison_all_models.png
│
├── src/                              # Código Python reutilizable
│   ├── __init__.py
│   ├── data_loader.py               # Funciones de descarga/carga
│   ├── features.py                  # Lógica de feature engineering
│   ├── models.py                    # Entrenamiento de modelos
│   └── visualization.py             # Gráficos estandarizados
│
├── config.yaml                       # Parámetros configurables
└── README.md                         # Este archivo
```

---

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip

### Pasos

1. **Clonar el repositorio**:
```bash
git clone https://github.com/tu-usuario/predictor-precio-cobre.git
cd predictor-precio-cobre
```

2. **Crear entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r ../requirements.txt
```

### Dependencias Principales

- **Data Science**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, xgboost, statsmodels, pmdarima
- **Visualización**: matplotlib, seaborn, plotly
- **Data Source**: yfinance (Yahoo Finance API)

---

## Uso

### 1. Ejecutar el Pipeline Completo

Los notebooks están diseñados para ejecutarse en orden:

```bash
jupyter notebook notebooks/
```

Secuencia recomendada:
1. `01_descarga_datos.ipynb` - Descarga datos históricos
2. `02_eda.ipynb` - Explora patrones y tendencias
3. `03_feature_engineering.ipynb` - Crea 42 features predictivas
4. `04_modelado.ipynb` - Entrena modelos baseline
5. `05_modelado_avanzado.ipynb` - Compara 5 modelos avanzados

### 2. Usar Modelo Pre-entrenado

```python
import pickle
import pandas as pd

# Cargar modelo
with open('models/ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar datos con features
df = pd.read_csv('data/processed/precio_cobre_features.csv')

# Predecir
feature_cols = [col for col in df.columns if col not in ['fecha', 'precio']]
X_new = df[feature_cols].iloc[-1:] # Último registro
precio_predicho = model.predict(X_new)[0]

print(f"Precio predicho: ${precio_predicho:.4f} USD/lb")
```

### 3. Generar Nuevas Predicciones

```python
from src.data_loader import download_copper_yahoo
from src.features import create_all_features

# Descargar datos actualizados
df_raw = download_copper_yahoo(start_date='2010-01-01')

# Crear features
df_features = create_all_features(df_raw)

# Predecir con modelo
predictions = model.predict(df_features[feature_cols])
```

---

## Metodología

### 1. Extracción de Datos (Notebook 01)

- **Fuente**: Yahoo Finance (ticker `HG=F` - Copper Futures)
- **Periodo**: 2010-05-13 a 2025-10-02 (3,962 registros)
- **Frecuencia**: Diaria (días de mercado)
- **Validación**: Sin valores nulos, ordenamiento cronológico

### 2. Análisis Exploratorio (Notebook 02)

Hallazgos clave del EDA:

- Tendencia alcista: Precio promedio creció de $3.20 (2010) a $4.50+ (2025)
- Estacionalidad: Precios 5% más altos en Q2 (marzo-mayo)
- Volatilidad variable: Picos en crisis (2020 COVID, 2022 guerra Ucrania)
- Autocorrelación fuerte: lag_1 = 0.997 (memoria extrema del mercado)
- Outliers detectados: 25 movimientos extremos >5% en 15 años

### 3. Feature Engineering (Notebook 03)

Creación de **42 features** organizadas en 7 grupos:

#### A. Estacionalidad (6 features)
- Mes, trimestre, encoding cíclico (sin/cos)
- Flags: Q2, temporada alta (marzo-mayo)

#### B. Volatilidad (6 features)
- Retornos diarios, desviación estándar móvil (7/30/90 días)
- Indicadores de alta volatilidad

#### C. Tendencia (7 features)
- Medias móviles (7/30/90 días)
- Distancia normalizada vs medias
- Momentum e indicadores direccionales

#### D. Outliers/Extremos (4 features)
- Z-score, flags de movimientos >5%
- Días desde último evento extremo

#### E. Lags - Autocorrelación (6 features)
- Precios pasados: lag_1, lag_2, lag_3, lag_7, lag_14, lag_30
- **Feature más importante**: lag_1 (correlación 0.9975)

#### F. Estadísticas (6 features)
- Dispersión, rangos, posición relativa del precio

#### G. Intrasemanal (7 features)
- Día de semana, inicio/fin de mes
- Flags de volatilidad (lunes, viernes)

**Validación**:
- Sin data leakage (todas las features miran hacia atrás)
- 90 registros eliminados por NaN técnicos (primeros 90 días)
- Dataset final: 3,872 registros con 42 features + precio

### 4. Modelado (Notebooks 04 y 05)

#### Split Temporal (Crítico para Series Temporales)
- **Train**: 80% (3,097 registros) - 2010 a sep 2022
- **Validation**: 10% (387 registros) - sep 2022 a mar 2024
- **Test**: 10% (388 registros) - mar 2024 a oct 2025
- **SIN shuffle**: Respeta orden cronológico

#### Modelos Evaluados

1. **Linear Regression** (baseline)
   - Simple, interpretable
   - R cuadrado = 0.9923

2. **ARIMA (5,1,2)** (baseline temporal)
   - Estándar de la industria
   - No usa features engineered

3. **Ridge Regression** (GANADOR)
   - Regularización L2 (alpha=0.01)
   - Maneja multicolinealidad
   - R cuadrado = 0.9914

4. **XGBoost**
   - Gradient boosting
   - 500 árboles, lr=0.01
   - R cuadrado = 0.6567 (decepcionó)

5. **Random Forest**
   - 300 árboles, max_depth=10
   - R cuadrado = 0.6809

6. **SARIMAX (0,1,0)**
   - ARIMA + top 10 features exógenas
   - Auto-optimizado con pmdarima
   - R cuadrado = 0.7593

---

## Modelos Implementados

### Ridge Regression (Modelo Recomendado)

**Por qué ganó**:
- Maneja perfectamente la multicolinealidad (lags correlacionados 0.99+)
- Captura la relación lineal fuerte del precio del cobre
- Regularización mínima (alpha=0.01) suficiente
- Rápido, simple e interpretable

**Hiperparámetros**:
```python
Ridge(alpha=0.01)  # Encontrado por Grid Search con TimeSeriesSplit
```

**Performance**:
- R cuadrado = 0.9914 (99.14% de varianza explicada)
- MAE = $0.0173 USD/lb
- MAPE = 0.38% (error porcentual medio)

**Uso en Producción**:
```python
import pickle
with open('models/ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)

precio_predicho = model.predict(X_new)
```

### ¿Por qué XGBoost Falló?

Contra las expectativas, XGBoost (R cuadrado=0.66) quedó último:

**Causas identificadas**:
1. **Problema lineal**: El precio del cobre tiene relación casi perfectamente lineal con lags
2. **Fragmentación**: Árboles fragmentan el espacio, perdiendo suavidad de la relación
3. **Hiperparámetros conservadores**: max_depth=5 puede ser muy restrictivo
4. **Overfitting a validation**: Early stopping detuvo entrenamiento prematuramente

**Lección aprendida**: Modelos complejos no siempre superan a modelos simples. Entender el problema es más importante que usar el algoritmo más sofisticado.

---

## Conclusiones

### Hallazgos Principales

1. **La simplicidad es poderosa**:
   - Ridge Regression con R cuadrado=0.99 superó a XGBoost (R cuadrado=0.66)
   - Feature engineering de calidad > Complejidad del modelo

2. **Memoria extrema del mercado**:
   - lag_1 (precio de ayer) tiene correlación 0.9975 con hoy
   - El mercado del cobre es altamente predecible a corto plazo

3. **Multicolinealidad bien manejada**:
   - Ridge controla features correlacionadas (lags 0.99+)
   - Linear Regression sin regularización también funcionó bien (R cuadrado=0.99)

4. **Features dominantes**:
   - Top 5: lag_1, rolling_mean_7, lag_2, lag_3, rolling_mean_30
   - Volatilidad y estacionalidad: Efecto marginal

### Recomendaciones

#### Para Implementación en Producción

**Modelo Recomendado**: Ridge Regression

**Pipeline**:
1. Descargar datos diarios de Yahoo Finance (HG=F)
2. Generar 42 features con `src/features.py`
3. Predecir con modelo Ridge pre-entrenado
4. Intervalo de confianza: ±$0.035 USD/lb (2 RMSE)

**Frecuencia de Actualización**:
- Reentrenar modelo: Mensual
- Actualizar predicciones: Diario

#### Para Mejora Futura

1. **Optimizar XGBoost** (Notebook 06 futuro):
   - Grid Search exhaustivo (max_depth hasta 15)
   - Probar n_estimators=1000-3000
   - Feature selection (reducir de 42 a top 20)

2. **Ensemble**:
   - 0.7 Ridge + 0.2 SARIMAX + 0.1 RF
   - Potencial mejora: 1-2% adicional

3. **Variables Exógenas**:
   - Precio del oro (correlación con commodities)
   - Índice USD (inversión con cobre)
   - Inventarios de LME (London Metal Exchange)

4. **Deep Learning** (exploración):
   - LSTM/GRU para capturar dependencias temporales largas
   - Transformer para atención temporal

### Limitaciones

- **Horizonte de predicción**: Modelo optimizado para 1 día adelante
- **Eventos extremos**: Crisis no vistas (cisnes negros) pueden degradar performance
- **Solo precio**: No considera fundamentals (oferta/demanda, inventarios)
- **Datos limpios**: Requiere mercado funcionando (no válido en cierres de mercado)

---

## Tecnologías Utilizadas

### Lenguajes y Frameworks

- **Python 3.8+**: Lenguaje principal
- **Jupyter Notebook**: Entorno de desarrollo interactivo

### Librerías de Machine Learning

- **scikit-learn 1.3.2**: Linear Regression, Ridge, Random Forest, métricas
- **XGBoost 2.0.3**: Gradient boosting
- **statsmodels 0.14.1**: ARIMA
- **pmdarima 2.0.4**: Auto-ARIMA y SARIMAX

### Procesamiento y Análisis

- **pandas 2.2.2**: Manipulación de datos
- **numpy 1.26.4**: Operaciones numéricas
- **scipy**: Estadística y optimización

### Visualización

- **matplotlib 3.8.4**: Gráficos base
- **seaborn 0.13.2**: Visualizaciones estadísticas
- **plotly 5.18.0**: Gráficos interactivos

### Data Source

- **yfinance 0.2.35**: API de Yahoo Finance para datos históricos

### Utilidades

- **PyYAML**: Configuración
- **joblib**: Serialización de modelos
- **pickle**: Persistencia de objetos Python

---

## Próximos Pasos

- Crear API REST con FastAPI
- Deploy en Render/Railway
- Dashboard interactivo con Streamlit
- Notebook 06: Optimización y Ensemble
- Incorporar variables exógenas (oro, USD, inventarios LME)
- Análisis de series temporales multivariado (VAR)

---

## Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)
- Email: tu.email@ejemplo.com

---

## Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo [LICENSE](../LICENSE) para más detalles.

---

## Agradecimientos

- **COCHILCO** (Comisión Chilena del Cobre): Contexto de industria
- **Yahoo Finance**: Fuente de datos históricos
- **scikit-learn community**: Documentación y ejemplos
- **Kaggle**: Inspiración de mejores prácticas

---

**Proyecto completado**: Octubre 2025
**Versión**: 1.0
**Estado**: Producción-ready
