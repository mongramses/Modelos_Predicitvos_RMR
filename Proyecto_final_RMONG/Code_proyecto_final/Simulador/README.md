# 📊 Predicción de Default de Crédito

Esta aplicación permite predecir la probabilidad de incumplimiento de pago de un cliente usando un modelo de regresión logística entrenado.

## Cómo usar

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecuta la app:

```bash
streamlit run app_prediccion_default.py
```

## Archivos

- `app_prediccion_default.py`: Código principal de Streamlit
- `logistic_model.pkl`: Modelo de regresión logística
- `scaler.pkl`: Escalador de variables
- `feature_names.pkl`: Columnas esperadas por el modelo
