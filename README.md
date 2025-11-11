# 🧪 QSAR Biodegradation MLflow Pipeline.
---
## Autores: Tobías Romero (2021214011) y Jenifer Roa (20222140xx)
### Pipeline modular para entrenamiento y tracking de modelos de clasificación binaria sobre el dataset QSAR Biodegradation (OpenML ID: 1494).
---

## ✨ Características

- Regresión Logística con grid search y registro manual en MLflow
- Red Neuronal (TensorFlow/Keras) con autologging
- Interpretación de resultados con Gemini
- Runs anidados para experimentación avanzada

## 📄 Instalación

```bash
pip install -r requirements.txt
```

## ⚙️ Configuración

Para habilitar la interpretación con Gemini, configure la variable de entorno:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Si no se configura, el pipeline continuará sin análisis LLM.

## 🧑‍💻 Uso

### Ejecución básica

```bash
python main.py
```

Ejecuta:
- Grid search de Regresión Logística (6 variantes)
- Entrenamiento de Red Neuronal
- Interpretación con Gemini

### Ejecución completa

```bash
python main.py --full
```

Incluye además:
- Runs anidados con experimentos adicionales
- Autologging global activado

## Resultados

Los experimentos se registran en MLflow bajo el nombre `qsar_biodeg_experiments`.

Artefactos generados:
- `confusion_matrix_logreg.png`
- `classification_report_logreg.json`
- `confusion_matrix_keras.png`
- `keras_config.txt`
- `gemini_insights.txt`

## 🧱 Estructura del proyecto
```
qsar_biodeg/
├── main.py
├── requirements.txt
├── README.md
└── src/
   └── qsar_biodeg/
      ├── config.py         # constantes, hiperparámetros, API key
      ├── data.py           # carga OpenML, split, escalado
      ├── metrics.py        # métricas y figuras (sin escribir a disco)
      ├── tracking.py       # helpers MLflow (runs, log_x, autolog, init)
      ├── pipeline.py       # orquestación
      ├── models/
      │  ├── logistic.py    # grid + logging manual
      │  └── keras_mlp.py   # Keras + autolog, métricas test
      └── llm/
         └── genai_client.py # Gemini (google genai)
```

## 📚 Referencias
- Dataset: [https://www.openml.org/d/1494](https://www.openml.org/d/1494)
- MLflow: [https://mlflow.org/docs/latest/quickstart.html](https://mlflow.org/docs/latest/quickstart.html) · [https://mlflow.org/docs/latest/tracking.html](https://mlflow.org/docs/latest/tracking.html)
- Keras: [https://www.tensorflow.org/guide/keras/sequential_model](https://www.tensorflow.org/guide/keras/sequential_model) · [https://www.tensorflow.org/api_docs/python/tf/keras/Model#save](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save)
- Google AI Studio / SDK `google genai` (Gemini)

