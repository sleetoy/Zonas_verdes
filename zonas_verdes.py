"""
Proyecto: Detección de zonas verdes en una imagen urbana.
Autor: Sebastián Leetoy Flores
Descripción:
Este programa analiza imagenes para detectar
las zonas verdes (vegetación).
Genera:
1. Una imagen de salida con las áreas verdes resaltadas.
2. Un archivo CSV con el porcentaje total de píxeles verdes.
"""

# --- Importación de librerías ---
import cv2          # Librería para procesamiento de imágenes
import numpy as np   # Librería para manejo de matrices y operaciones numéricas
import pandas as pd  # Librería para exportar resultados a CSV


# --- Configuración de rutas y nombres de archivos ---
imagen_entrada = "parque.png"           # Nombre de la imagen a analizar
imagen_salida = "parque_verde.png"      # Imagen resultante con zonas verdes resaltadas
archivo_csv = "resultado_parque.csv"    # Archivo CSV de salida


# --- Lectura de la imagen ---
img = cv2.imread(imagen_entrada)


# --- Conversión de color ---
# Convertimos de BGR (formato por defecto en OpenCV) a HSV,
# ya que este espacio de color facilita la detección de tonos específicos.
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# --- Definición del rango de color verde ---
# El rango se puede ajustar según el tipo de vegetación o luz.
verde_bajo = np.array([35, 40, 40])    # Límite inferior (tono, saturación, valor)
verde_alto = np.array([85, 255, 255])  # Límite superior


# --- Creación de la máscara ---
# Genera una imagen binaria donde los píxeles dentro del rango verde son blancos (255)
mascara = cv2.inRange(hsv, verde_bajo, verde_alto)


# --- Limpieza de ruido ---
# Se aplican operaciones morfológicas para eliminar imperfecciones pequeñas.
mascara = cv2.dilate(mascara, None, iterations=2)
mascara = cv2.erode(mascara, None, iterations=2)


# --- Cálculo del porcentaje de zona verde ---
total_pixeles = img.shape[0] * img.shape[1]   # Total de píxeles de la imagen
pixeles_verdes = cv2.countNonZero(mascara)    # Píxeles detectados como verdes
porcentaje_verde = (pixeles_verdes / total_pixeles) * 100


# --- Creación de imagen de salida ---
# Se copia la imagen original y se colorean las zonas verdes detectadas
salida = img.copy()
salida[mascara > 0] = [0, 255, 0]  # Color verde brillante en formato BGR


# --- Guardado de resultados ---
# Guarda la imagen con las zonas verdes resaltadas
cv2.imwrite(imagen_salida, salida)

# Crea un archivo CSV con los resultados
resultado = pd.DataFrame([{
    "imagen": imagen_entrada,
    "porcentaje_verde": round(porcentaje_verde, 2)
}])

resultado.to_csv(archivo_csv, index=False)


# --- Mensajes de confirmación ---
print("Procesamiento completado:")
print(f"   Imagen de salida: {imagen_salida}")
print(f"   CSV generado: {archivo_csv}")
print(f"   Porcentaje verde: {porcentaje_verde:.2f}%")

