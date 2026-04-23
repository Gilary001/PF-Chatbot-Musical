# PF-Chatbot-Musical

---
##  Chatbot Musical Inteligente


### Proyecto Final: Minería de Textos
**Institución:** Colegio Universitario de Cartago (CUC)  
**Profesor:** Osvaldo Gonzalez Chaves  

---

## 📋 Descripción

**HistoryBot** es un chatbot especializado en historia musical que utiliza una arquitectura de **Generación Aumentada por Recuperación (RAG)**. Su objetivo es analizar la evolución de las letras de canciones en inglés desde la década de 1990 hasta la de 2020. 

El sistema combina un clasificador de décadas fine-tuneado con una búsqueda semántica vectorial para ofrecer respuestas basándose exclusivamente en un corpus almacenado en **MongoDB**.


---
output: github_document
---

## 🚀 Instalación y Configuración del Entorno

Para garantizar el correcto funcionamiento del chatbot es necesario instalar las librerías de procesamiento de lenguaje natural, gestión de bases de datos y herramientas de visualización.

### 1. Dependencias del Sistema
Ejecute el siguiente bloque para instalar los paquetes necesarios:

```{python, eval=FALSE}
git clone https://github.com/Gilary001/PF-Chatbot-Musical.git

!pip install transformers datasets accelerate torch scikit-learn pymongo pandas numpy matplotlib seaborn
# Instalación de FAISS para búsqueda vectorial
# pip install faiss-cpu
```

## Estructura
* **`app/`**: Interfaz de usuario y servicios API.
    * `chatbot_app.py`: chatbot con Flan-t5
    * `Api_dh.py`: chatbot con API
* **`data/`**: Gestión del corpus original desde MongoDB(`corpus_canciones.py`).
* **`models/`**: Almacenamiento del **Clasificador de Décadas**.
    * `checkpoint-234/`: Pesos del modelo (`model.safetensors`), tokenizadores y estados del entrenamiento.
* **`notebooks/`**: Pipeline completo de experimentación.
    * Incluye el entrenamiento del clasificador, el desarrollo del RAG y los archivos de persistencia (`.pkl` y `.bin`) para los índices A y B.
* **`src/`**: Lógica modular del sistema.
    * `Api_chatbot_engine.py`: Cerebro del asistente para el chatbot con Api.
    * `chatbot_engine.py`: Cerebro del asistente para el chatbot con flan.
    * `rag_utils.py`: Funciones de búsqueda y procesamiento de embeddings.
    * `finetuning_utils.py`: Utilidades para el re-entrenamiento del modelo.
* **`resultados/`**: Evidencia de métricas, incluyendo la **Matriz de Confusión** del modelo fine-tuneado.

## 👥 Equipo de Trabajo

| Nombre | GitHub |
| :--- | :--- |
| **Gilary Granados** | [@Gilary001](https://github.com/Gilary001) |
| **Wedell Orozco** | [@wedellXrozcoG](https://github.com/wedellXrozcoG) |