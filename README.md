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

### 1.1 La configuración del proceso de adquisición

Esta etapa abarca la preparación del sistema de adquisición ultrasónica, incluyendo los scripts necesarios para la comunicación y calibración del robot colaborativo encargado de posicionar el transductor durante la toma de datos.

### 1.2 El procesamiento de los datos crudos

Esta etapa contempla el procesamiento de las señales adquiridas, lo cual incluye los scripts de visualización, el criterio utilizado para el etiquetado automático de los datos, el formato de almacenamiento adoptado y el flujo completo de preprocesamiento.

### 1.3 El entrenamiento de una red convolucional

Esta etapa implica los criterios de entrenamiento y validación aplicados a la red convolucional, diseñada específicamente para la detección precisa de tiempos de vuelo en los datos ultrasónicos.

## 2. Instrumentación Requerida

### 2.1 Sistema de adquisición

Se utilizaron dos tipos de transductores matriciales: 1) Un transductor Imasonic de
11×11 elementos. 2) Un transductor Doppler de 16×16 elementos, del cual se emple´o so-
lamente un subconjunto activo de 8×16 elementos para la emisi´on y recepci´on de se˜nales.
El sistema de adquisici´on empleado es un equipo multicanal con 128 canales en par-
alelo, marca SITAU, fabricado por la empresa Dasel S.L. (España).

Para la ejecuci´on de trayectorias precisas sobre las piezas de ensayo, se utiliz´o un
brazo robot colaborativo de seis ejes (6 grados de libertad), modelo UR10e, fabricado por
Universal Robots (Dinamarca).

El conjunto de pruebas se realiz´o sobre seis piezas de geometr´ıa diversa, seleccionadas
con el objetivo de proporcionar un desaf´ıo adecuado para evaluar el desempe˜no de las
redes neuronales en etapas posteriore

## Documentación
Se adjunta también un trabajo de fin de máster (UPM) que explica con mayor detalle puntos teóricos y prácticos:

[Descargar TFM asociado.pdf](https://github.com/marceloLarreaxx/tutorial-de-datos-ultrasonicos/raw/main/Documentacion/Memoria_TFM.pdf)
