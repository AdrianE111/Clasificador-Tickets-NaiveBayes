# TicketClassify - Clasificador de Tickets con Naïve Bayes

## Estructura del Proyecto

```
Clasificador-Tickets-NaiveBayes/
│
├── app.py                          # Servidor Flask (backend)
├── naive_bayes.py                  # Implementación del algoritmo
├── preprocess.py                   # Preprocesamiento de texto
├── evaluate.py                     # K-Folds y evaluación
│
├── templates/
│   └── index.html                  # Interfaz web completa (frontend)
│
├── data/
│   └── archive/
│       ├── customer_support_tickets.csv  # Archivo vacío (placeholder para cargar Bitext)
│       ├── customer_support_tickets_mejorado.csv  # Dataset mejorado en inglés (opcional)
│       └── customer_support_tickets_mejorado_es.csv  # Dataset mejorado en español (opcional)
│
├── modelo_entrenado.pkl            # Modelo persistente
├── requirements.txt                # Dependencias Python
├── .gitignore                      # Archivos a ignorar en Git
└── README.md                       # Este archivo
```

---

## Instalación

### Requisitos Previos
- Python 3.8+
- pip

### Paso 1: Crear Virtual Environment
```bash
python -m venv .venv
```

### Paso 2: Activar Virtual Environment

**En PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**En Command Prompt (cmd):**
```cmd
.\.venv\Scripts\activate.bat
```

**En Linux/Mac:**
```bash
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 4: Descargar Recursos NLTK
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## Uso

### 1. Entrenar el Modelo (Si No Existe)

```bash
python naive_bayes.py
```

> El script `naive_bayes.py` carga automáticamente el dataset recomendado de Bitext desde Hugging Face cuando encuentra `data/archive/customer_support_tickets.csv`. Si hay problemas de conexión, puede usar los datasets mejorados locales.

### 2. Evaluar con K-Folds

```bash
python evaluate.py
```

### 3. Ejecutar la Aplicación Web

```bash
python app.py
```

Abre: `http://localhost:5000`

---

## Descripción del Código

### naive_bayes.py
- **Clase:** `NaiveBayesClassifier`
- **Métodos:** `train()`, `predict()`, `predict_proba()`, `save_model()`, `load_model()`
- **Técnicas:** Laplace Smoothing, Log-Probabilidades, Normalización

### preprocess.py
- `clean_text(text)`: Limpia, tokeniza y lematiza
- `build_vocabulary(corpus_tokens)`: Crea Bag of Words
- `load_and_preprocess_data(csv_path)`: Carga y mapea datos

### evaluate.py
- **Clase:** `ModelEvaluator`
- K-Folds Cross Validation manual
- Métricas y Matriz de Confusión

### app.py
- Servidor Flask
- Endpoints: `/predict`, `/health`
- Carga automática del modelo

---

## Análisis del Modelo

### Dataset
- **Fuente:** Bitext Customer Support LLM Chatbot Training Dataset (Hugging Face)
- **URL:** https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- **Total:** 26,872 tickets
- **Categorías originales:** 11 (ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION)
- **Mapeo a 5 categorías:** Soporte Técnico, Facturación, Consulta General, Queja, Cancelación
- **Vocabulario:** ~10,000+ palabras únicas (estimado)
- **Lenguaje:** Inglés
- **Etiquetas:** Consistentes y verificadas (generadas de forma controlada)

### Performance Esperado
- **Accuracy Global:** 75-85% (con dataset de 26k+ instancias)
- **Macro F1:** 0.75-0.85

### Consideraciones
El performance depende de la calidad y discriminabilidad de las palabras en el dataset.

---

## Dependencias

- pandas >= 1.3.0
- numpy >= 1.21.0
- nltk >= 3.6.0
- datasets >= 2.0.0
- flask >= 2.0.0
- matplotlib >= 3.4.0 (opcional)
- seaborn >= 0.11.0 (opcional)

---

## Autores

- **Adrián Escalón**
- **Pablo Bocel**
- **Alejandro Estupinián**

**Universidad:** Rafael Landívar
**Curso:** Inteligencia Artificial

---

## Referencias

- Dataset: [Bitext Customer Support LLM Chatbot Training Dataset - Hugging Face](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- NLTK Documentation
- Naïve Bayes Algorithm
