# TicketClassify - Clasificador de Tickets con Naïve Bayes

## Descripción del Proyecto

TicketClassify es un **sistema de clasificación automática de tickets de soporte técnico** basado en el algoritmo **Naïve Bayes Multinomial**. El proyecto implementa desde cero cada componente del algoritmo: preprocesamiento de texto, construcción del vocabulario (Bag of Words), cálculo de probabilidades con Laplace Smoothing, evaluación con K-Folds Cross Validation y una interfaz web funcional.

**Objetivo:** Clasificar automáticamente solicitudes de clientes en 5 categorías:
- Soporte Técnico
- Facturación
- Consulta General
- Queja
- Cancelación

---

## Requisitos del Proyecto

### Requisitos Técnicos Cumplidos

✅ **Implementación Manual de Naïve Bayes**
- Algoritmo implementado desde cero (sin scikit-learn)
- Cálculo manual de probabilidades a priori P(c)
- Cálculo de verosimilitud con Laplace Smoothing
- Uso de logaritmos para evitar underflow numérico

✅ **Técnicas Obligatorias**
- Bag of Words: Construcción del vocabulario
- Laplace Smoothing: (count + 1) / (total + vocab_size)
- Suma de Logaritmos: Productos → log(P) para estabilidad numérica
- K-Folds Cross Validation: K=5, implementación manual
- Matriz de Confusión: Análisis de confusiones entre clases
- Métricas por clase: Precision, Recall, F1-Score, Accuracy, Macro F1

✅ **Preprocesamiento de Texto**
- Tokenización con NLTK
- Eliminación de stopwords (inglés)
- Lematización con WordNetLemmatizer
- Conversión a minúsculas y limpieza de caracteres especiales

✅ **Persistencia del Modelo**
- Guardado en pickle: `modelo_entrenado.pkl`
- Carga automática en la aplicación web

✅ **Interfaz Web Completa**
- Sistema de tickets con ID autogenerado
- Campo de asunto y descripción detallada
- Historial persistente con localStorage
- Modal de detalles para cada ticket
- Tema claro/oscuro
- Ejemplos rápidos para guiar al usuario
- Visualización de probabilidades por clase

---

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
│       └── customer_support_tickets.csv  # Dataset
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

### Paso 1: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Descargar Recursos NLTK
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## Uso

### 1. Entrenar el Modelo (Si No Existe)

```bash
python naive_bayes.py
```

### 2. Evaluar con K-Folds

```bash
python evaluate.py
```

### 3. Ejecutar la Aplicación Web

```bash
python app.py
```

Abre: `http://localhost:5000`

### 4. Diagnosticar el Modelo

```bash
python diagnostico.py
```

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
- **Total:** 8,469 tickets
- **Clases:** 5 categorías balanceadas (19.3% - 20.7% cada una)
- **Vocabulario:** 6,266 palabras únicas

### Performance Esperado
- **Accuracy Global:** 65-75%
- **Macro F1:** 0.65-0.75

### Consideraciones
El performance depende de la calidad y discriminabilidad de las palabras en el dataset.

---

## Dependencias

- pandas >= 1.3.0
- numpy >= 1.21.0
- nltk >= 3.6.0
- flask >= 2.0.0
- matplotlib >= 3.4.0 (opcional)
- seaborn >= 0.11.0 (opcional)

---

## Autores

- **Pablo Bocel**
- **Adrián Escalón**
- **Alejandro Estupinián**

**Universidad:** Rafael Landívar
**Curso:** Inteligencia Artificial

---

## Referencias

- Dataset: [Customer Support Ticket Dataset - Kaggle](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
- NLTK Documentation
- Naïve Bayes Algorithm
