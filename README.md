# Tutorial de datos ultrasonicos

Este documento tiene como objetivo guiar paso a paso el proceso completo de adquisición, procesamiento y análisis de datos ultrasónicos utilizando transductores de tipo matricial, con el propósito de entrenar modelos de redes neuronales convolucionales (CNN). A lo largo del documento, se detallará cómo configurar el sistema de adquisición, cómo procesar y etiquetar los datos obtenidos, y cómo utilizar dichos datos para el entrenamiento de modelos de inteligencia artificial orientados a la detección de tiempos de vuelo en las imágenes ultrasónicas. Además, se explicará cómo ejecutar los scripts desarrollados, con ejemplos prácticos que permitirán la replicación del flujo completo desde los datos crudos hasta la inferencia 


## Estructura del proyecto
- `Calibracion_sistema:`
  Scripts para realizar la calibración del sistema de adquisición ultrasónica.
- `Adquisicion_datos:`
  Código para adquirir señales crudas de sensores ultrasónicos
- `Post_procesamiento:`
  Procesamiento de datos, creación de formatos .pickle, etiquetado.
- `Entrenamiento_cnn:`
  Preparación de datos y entrenamiento del modelo de red neuronal convolucional.

## Cómo utilizar este repositorio
  
Para trabajar con este proyecto:

1. **Clona este repositorio**:
   Este comando descarga el proyecto completo a tu ordenador:
   
   ```bash
   git clone https://github.com/marceloLarreaxx/tutorial-de-datos-ultrasonicos.git
   
2. **Accede a la carpeta del proyecto**:
   En tu terminal
   
   ```bash
   cd tutorial-de-datos-ultrasonicos

## Requisitos
Para la ejecución correcta de los scripts, asegurarse de tener las siguientes versiones instaladas:
- Python: 3.7-3.10
- TensorFlow: 2.10

## 1. Flujo de Trabajo
En términos generales, el flujo de actividades se divide en tres etapas principales:

### 1.1 ggg

## Documentación
Se adjunta también un trabajo de fin de máster (UPM) que explica con mayor detalle puntos teóricos y prácticos:

[Descargar TFM asociado.pdf](https://github.com/marceloLarreaxx/tutorial-de-datos-ultrasonicos/raw/main/Documentacion/Memoria_TFM.pdf)
