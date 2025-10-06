# Proyecto: Predictor de Precio del Cobre

## Objetivo del Proyecto

Crear un sistema completo de Machine Learning para predecir el precio del cobre chileno utilizando series temporales y técnicas de feature engineering.

### Contexto de Negocio
- **Relevancia:** Chile es el mayor productor de cobre del mundo
- **Uso empresarial:** Planificación presupuestaria, cobertura de riesgo financiero, decisiones de inversión
- **Complejidad:** Fácil - Ideal para comenzar con series temporales
- **Datos:** Públicos (COCHILCO y Yahoo Finance)

### Problema a Resolver
**Pregunta clave:** ¿Cuál será el precio del cobre mañana/próxima semana/próximo mes?

## Modelos a Implementar

### Fase 1: Modelos Baseline (COMPLETADO)

1. **Linear Regression** (baseline con features engineered)
   - Modelo simple para tener una referencia
   - Usa features engineered (lags, rolling stats)
   - Fácil de interpretar

2. **ARIMA** (modelo especializado en series temporales)
   - Captura tendencia y estacionalidad automáticamente
   - Estándar de la industria para series temporales
   - Orden: (p=5, d=1, q=2)

### Fase 2: Modelos Avanzados (EXPANSION)

3. **Ridge Regression** (regularización L2)
   - Soluciona problema de multicolinealidad detectado en notebook 03
   - Penaliza coeficientes grandes para evitar overfitting
   - Grid search para encontrar mejor alpha (0.01, 0.1, 1, 10, 100)
   - Esperado: Mejor que Linear Regression por manejo de features correlacionadas

4. **XGBoost** (gradient boosting)
   - Modelo robusto que maneja multicolinealidad sin problema
   - Captura no-linealidades e interacciones automáticamente
   - Hiperparámetros: n_estimators=500, learning_rate=0.01, max_depth=5
   - Feature importance automático
   - Esperado: Mejor modelo del proyecto (R² > 0.95)

5. **Random Forest** (ensemble de árboles)
   - Alternativa robusta a XGBoost
   - Menos propenso a overfitting
   - Hiperparámetros: n_estimators=300, max_depth=10, min_samples_split=5
   - Bueno para identificar features importantes

6. **SARIMAX** (ARIMA con variables exógenas)
   - Evolución de ARIMA que aprovecha las 42 features creadas
   - Usa top 5-10 features más correlacionadas como exógenas
   - Auto-ARIMA para encontrar orden óptimo automáticamente
   - Esperado: Mejor que ARIMA simple

### Fase 3: Optimización y Ensemble (OPCIONAL)

7. **Feature Selection**
   - Reducir de 42 a top 15-20 features (eliminar redundancia)
   - Técnicas: Recursive Feature Elimination, LASSO, importancia de XGBoost
   - Objetivo: Mejorar interpretabilidad y reducir overfitting

8. **Ensemble (Promedio Ponderado)**
   - Combinar predicciones de mejores 3-4 modelos
   - Pesos basados en performance en validation set
   - Típicamente reduce error 2-5% adicional
   - Ejemplo: 0.4*XGBoost + 0.3*Ridge + 0.2*SARIMAX + 0.1*RF

9. **Walk-Forward Validation**
   - Validación en múltiples ventanas temporales (no solo un test set)
   - Simula predicción en producción más realísticamente
   - Evalúa estabilidad del modelo a través del tiempo

## Estructura del Proyecto

```
predictor-precio-cobre/
├── data/
│   ├── raw/                    # CSV/Excel descargados de fuentes
│   └── processed/              # Datos con features creados
│
├── docs/                       # Documentación del proyecto
│   ├── PRESENTACION.md        # Presentación ejecutiva (15 slides)
│   ├── claude.md              # Este archivo - Plan del proyecto
│   ├── TASKS.md               # Checklist de tareas completadas
│   └── intrucciones_preciocobre.md  # Instrucciones iniciales
│
├── notebooks/                  # Jupyter Notebooks para exploración
│   ├── 01_descarga_datos.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelado.ipynb
│   ├── 05_modelado_avanzado.ipynb        # COMPLETADO: Ridge, XGBoost, RF, SARIMAX
│   └── 06_ensemble_optimizacion.ipynb    # OPCIONAL: Ensemble y tuning
│
├── src/                        # Código Python reutilizable
│   ├── __init__.py
│   ├── data_loader.py         # Descarga y carga de datos
│   ├── features.py            # Feature engineering (42 features)
│   ├── models.py              # 6 modelos: LR, Ridge, XGBoost, RF, ARIMA, SARIMAX
│   └── visualization.py       # Gráficos estandarizados
│
├── models/                     # Modelos .pkl entrenados
├── reports/figures/            # Gráficos PNG
└── config.yaml                 # Parámetros configurables
```

## Features Principales

### Variables de Entrada
1. **Precio histórico del cobre** (variable principal)
2. **Tendencia temporal:** ¿El precio está subiendo o bajando?
3. **Estacionalidad:** Patrones que se repiten en ciertos meses/trimestres
4. **Promedios móviles:** Media de últimos 7, 30, 90 días
5. **Volatilidad:** Variabilidad reciente del precio

### Lags (valores pasados)
- `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_30`
- Ejemplo: `lag_1` = precio de ayer, `lag_7` = precio de hace una semana

### Promedios Móviles (Rolling Statistics)
- `rolling_mean_7/30/90`: Promedios móviles
- `rolling_std_7/30/90`: Desviación estándar móvil
- `rolling_min_7/30/90` y `rolling_max_7/30/90`
- Captura tendencias de corto, mediano y largo plazo

### Features Temporales
- `year`, `month`, `day`, `dayofweek`, `quarter`
- `is_month_start`, `is_month_end` (binarios)
- Captura estacionalidad

### Features de Tendencia
- `price_diff`: Cambio día a día (precio_hoy - precio_ayer)
- `price_pct_change`: Retorno porcentual (%)
- `volatility_7d`, `volatility_30d`: Desviación estándar de retornos

## Métricas de Éxito Esperadas

### Fase 1: Modelos Baseline

#### Linear Regression
- **R²**: 0.85-0.92 (explica 85-92% de varianza)
- **MAE**: < $0.15 USD/lb (Mean Absolute Error)
- **RMSE**: < $0.20 USD/lb (Root Mean Squared Error)

#### ARIMA
- Similar o ligeramente mejor que Linear Regression
- Mejor captura de tendencias a corto plazo

### Fase 2: Modelos Avanzados (Objetivos)

#### Ridge Regression
- **R²**: 0.90-0.94 (mejor que Linear Regression)
- **MAE**: < $0.12 USD/lb
- **RMSE**: < $0.16 USD/lb
- Mejora esperada: 5-10% vs Linear Regression

#### XGBoost (Mejor modelo esperado)
- **R²**: 0.95-0.98
- **MAE**: < $0.08 USD/lb
- **RMSE**: < $0.12 USD/lb
- Mejora esperada: 15-25% vs Linear Regression

#### Random Forest
- **R²**: 0.93-0.96
- **MAE**: < $0.10 USD/lb
- **RMSE**: < $0.14 USD/lb

#### SARIMAX
- **R²**: 0.90-0.95
- Mejor que ARIMA por uso de variables exógenas
- Mejora esperada: 5-15% vs ARIMA

### Explicación de Métricas
- **MAE**: Error promedio en dólares
- **RMSE**: Penaliza errores grandes
- **R²**: % de varianza explicada (0-1)
- **MAPE**: Error porcentual medio

## Dependencias Principales

### Fase 1 (Completado)
- pandas, numpy
- scikit-learn
- statsmodels (ARIMA)
- matplotlib, seaborn, plotly
- yfinance (fuente de datos)
- pyyaml
- jupyter

### Fase 2 (Expansión)
- xgboost (gradient boosting)
- pmdarima (auto_arima para SARIMAX)
- scipy (optimización para ensemble)

## Extracción de Datos

### Fuente Principal: Yahoo Finance (yfinance)
**Ticker:** `HG=F` (Copper Futures)

**Función principal:** `download_copper_yahoo(start_date, end_date)`
- Descarga datos desde 2010-01-01 hasta hoy
- Retorna DataFrame con: `['fecha', 'precio_cobre_usd_lb']`
- Guarda en `data/raw/precio_cobre_yahoo.csv`

**Función auxiliar:** `load_copper_data(source='yahoo')`
- Carga datos ya descargados
- Parsea fechas correctamente
- Elimina valores nulos
- Ordena cronológicamente

**Alternativa:** Descarga manual desde COCHILCO si Yahoo Finance falla

## Pipeline

```
Datos brutos → Feature Engineering → Split Train/Val/Test →
→ Entrenar modelos → Evaluar → Comparar → Seleccionar mejor → Deploy
```

## Flujo de Trabajo en Notebooks

### 1. 01_descarga_datos.ipynb
- Descargar datos de Yahoo Finance (ticker: HG=F)
- Desde 2010-01-01 hasta hoy
- Verificar missing values
- Guardar en `data/raw/precio_cobre_yahoo.csv`

### 2. 02_eda.ipynb (Análisis Exploratorio)
- Gráfico de serie temporal completa (2010-2024)
- Distribución de precios (histograma)
- Boxplot por año para ver tendencia
- Análisis de estacionalidad (precio promedio por mes)
- Autocorrelación (ACF/PACF plots) - determina orden ARIMA
- Rolling statistics (media y std móviles)
- Identificar outliers y eventos extremos

### 3. 03_feature_engineering.ipynb
- Crear lags, rolling features, temporales, tendencia
- Visualizar correlación (heatmap)
- Análisis de importancia de features
- Verificar multicolinealidad (VIF)
- Guardar `precio_cobre_processed.csv`

### 4. 04_modelado.ipynb (COMPLETADO)
- Split temporal (train/val/test) - **SIN shuffle**
- Entrenar Linear Regression y ARIMA
- Predicciones en test set
- Calcular métricas (MAE, RMSE, R², MAPE)
- Gráficos de comparación: real vs predicciones
- Análisis de residuales
- Conclusión: ¿Qué modelo es mejor?
- Guardar mejor modelo en `models/`

### 5. 05_modelado_avanzado.ipynb (NUEVO)
- Cargar datos procesados y split del notebook 04
- **Ridge Regression con Grid Search**
  - Probar alphas: [0.01, 0.1, 1, 10, 100]
  - Validación cruzada temporal (TimeSeriesSplit)
  - Comparar con Linear Regression
- **XGBoost**
  - Configuración inicial: n_estimators=500, lr=0.01, max_depth=5
  - Feature importance plot (top 20 features)
  - Análisis de predicciones
- **Random Forest**
  - n_estimators=300, max_depth=10
  - Comparar feature importance con XGBoost
- **SARIMAX (Auto-ARIMA con exógenas)**
  - Seleccionar top 10 features más correlacionadas
  - Usar pmdarima.auto_arima para encontrar orden óptimo
  - Comparar con ARIMA simple
- **Tabla comparativa final**: Todos los modelos (6 total)
- **Gráfico de comparación**: Real vs predicciones de todos los modelos
- Guardar mejores modelos en `models/`

### 6. 06_ensemble_optimizacion.ipynb (OPCIONAL)
- Feature selection con RFE o importancia de XGBoost
- Reentrenar modelos con features seleccionadas
- Crear ensemble ponderado de top 3 modelos
- Walk-forward validation
- Análisis de errores por periodo temporal
- Conclusiones finales y recomendaciones

## Objetivo Personal

Proyecto #1 de portfolio de 30 proyectos ML/DL. Debe ser:
- Profesional (para mostrar en entrevistas)
- Educativo (con explicaciones claras)
- Reproducible (ejecutable por cualquiera)
- Extensible (fácil agregar más features/modelos)

## Notas de Implementación

### Estilo de Código
- Docstrings en todas las funciones (Google style)
- Type hints donde sea apropiado
- Nombres de variables en español (contexto chileno)
- Comentarios explicativos en secciones complejas
- Prints informativos para tracking de progreso
- **SIN EMOJIS**: No usar emojis en código, comentarios, prints, ni documentación

### Parámetros Clave
- **Split temporal sin shuffle** (crítico en series temporales)
- Orden ARIMA: (p=5, d=1, q=2)
  - p=5: Usa últimos 5 valores (autorregresivo)
  - d=1: Diferenciación de primer orden (estacionariedad)
  - q=2: Media móvil de últimos 2 errores
- Split de datos: 80% train, 10% val, 10% test
- Fuente de datos: Yahoo Finance (yfinance) como principal

### Archivos Clave del Proyecto
- `src/data_loader.py`: Funciones para descargar/cargar datos
- `src/features.py`: Lógica de feature engineering
- `src/models.py`: Entrenamiento de modelos
- `src/visualization.py`: Gráficos estandarizados
- `config.yaml`: Parámetros configurables centralizados

### Importante
- Código ejecutable inmediatamente después de `pip install -r requirements.txt`
- Manejo básico de errores (try/except)
- Notebooks auto-explicativos para aprendizaje ML
- Proyecto reproducible y extensible
