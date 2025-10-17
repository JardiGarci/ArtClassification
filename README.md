# 🎨 Clasificación de Movimientos Artísticos mediante Análisis Multifractal y Topológico

Este repositorio contiene los códigos, datos y referencias asociados al proyecto **“Clasificación de movimientos artísticos mediante análisis multifractal y análisis topológico”**.  
El objetivo principal es explorar las diferencias estructurales entre distintos estilos pictóricos a través de herramientas de la teoría de sistemas complejos y el análisis de datos.  

## 🧠 Descripción del Proyecto

El estudio del arte desde una perspectiva cuantitativa busca caracterizar y clasificar obras pictóricas mediante métricas matemáticas.  
En este proyecto se implementan técnicas como:

- **Análisis multifractal (MF-DFA 2D)** para estimar la complejidad estructural en las imágenes.  
- **Análisis topológico** basado en la persistencia homológica para identificar patrones de conectividad.  
- **Clasificadores supervisados** (SVM, k-NN, Random Forest) para discriminar entre movimientos artísticos.  

La base de datos incluye pinturas de los movimientos **Barroco**, **Post Impresionismo** y **Cubismo**, enfocadas en el género **retratos**, con 200 pinturas por cada estilo.

## 📂 Contenido del Repositorio

- `src/` → Scripts en Python para procesamiento, análisis y clasificación.  
- `notebooks/` → Experimentos en Jupyter Notebooks.  
- `data/` → Muestras de imágenes organizadas por movimiento artístico.  
- `results/` → Gráficas, espectros multifractales y resultados de clasificación.  
- `docs/` → Referencias, reportes y material complementario.  

## ⚙️ Requisitos

- Python 3.10+
- Descargar repositorio SNIC : https://github.com/achanta/SNIC.git
- Bibliotecas principales: `numpy`, `scipy`, `opencv-python`, `matplotlib`, `scikit-learn`  
