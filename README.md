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

Se utilizaron dos tipos de transductores matriciales: 1) Un transductor Imasonic de 11×11 elementos. 2) Un transductor Doppler de 16×16 elementos, del cual se empleó solamente un subconjunto activo de 8×16 elementos para la emisión y recepción de señales. El sistema de adquisici´on empleado es un equipo multicanal con 128 canales en paralelo, marca SITAU, fabricado por la empresa Dasel S.L. (España).

Para la ejecución de trayectorias precisas sobre las piezas de ensayo, se utilizó un brazo robot colaborativo de seis ejes (6 grados de libertad), modelo UR10e, fabricado por Universal Robots (Dinamarca).

El conjunto de pruebas se realizó sobre seis piezas de geometría diversa, seleccionadas con el objetivo de proporcionar un desafío adecuado para evaluar el desempeño de las redes neuronales en etapas posteriores (Figura 2.1)

<div align="center">
<img src="Imagenes/fig_2_1.png" alt="Piezas de referencia" width="500" />
</div>

Toda la implementación de código, incluyendo las interfaces gráficas desarrolladas para esta etapa, se realizó en Python, utilizando como entorno de desarrollo integrado PyCharm.

A continuación, se describe el setup experimental con mayor detalle.

## Documentación
Se adjunta también un trabajo de fin de máster (UPM) que explica con mayor detalle puntos teóricos y prácticos:

## 3. Adquisición de Datos

### 3.1 Calibración del Sistema

El primer paso en el proceso de adquisición consiste en la calibración del sistema, cuyo objetivo es determinar con precisión la posición y orientación del transductor respecto a cada pieza evaluada.

Para ello, se comienza ajustando el Punto Central de la Herramienta (PCH) del brazo robótico. Este punto se define sobre el elemento central del transductor matricial y es fundamental, ya que todos los movimientos y trayectorias posteriores del robot se calcularán con base en dicha referencia.

La calibración implica una secuencia predefinida de inclinaciones del transductor, ejecutadas mediante el brazo robótico. En cada una de estas posiciones, se adquieren datos de tiempo de vuelo (TOF). El procedimiento compara los tiempos de vuelo teóricos, calculados a partir de modelos geométricos, con los valores medidos durante la adquisición.

Con esta información, se aplica un análisis de regresión por mínimos cuadrados que permite estimar las correcciones necesarias en las coordenadas del PCH, de modo que este quede correctamente alineado con el centro real del transductor.

A continuación, se detallan los pasos específicos para llevar a cabo este proceso de calibración:

#### 3.1.1 Interfaz Gráfica

La interfaz principal utilizada para este proceso se muestra a continuación. Esta fue implementada en el script [alinear_app_2.py](Calibracion_sistema/alinear_app_2.py). Para llevar a cabo la calibración descrita anteriormente, se siguen los siguientes pasos:

## Documentación

Se adjunta también un trabajo de fin de máster (UPM) que explica con mayor detalle puntos teóricos y prácticos:

[Descargar TFM asociado.pdf](https://github.com/marceloLarreaxx/tutorial-de-datos-ultrasonicos/raw/main/Documentacion/Memoria_TFM.pdf)
