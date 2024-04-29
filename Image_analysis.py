import cv2
import numpy as np
import matplotlib.pyplot as plt

# Realizar un conteo de árboles de una imágen aérea

image = cv2.imread('imagen_aerea2.JPG')## Cargar la imagen

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)## Convertir la imagen a HSV

h, s, v = cv2.split(hsv_image)## Dividir la imagen HSV en sus componentes

plt.figure(figsize=(12, 4))## Visualizar las matrices

plt.subplot(1, 3, 1)## Visualizar la matriz de tono (H)
plt.imshow(h, cmap='hsv', vmin=0, vmax=179)  ## El tono tiene un rango de 0 a 179 en OpenCV
plt.title('Hue')
plt.colorbar()

plt.subplot(1, 3, 2)## Visualizar la matriz de saturación (S)
plt.imshow(s, cmap='gray', vmin=0, vmax=255)  ## La saturación tiene un rango de 0 a 255 en OpenCV
plt.title('Saturation')
plt.colorbar()

plt.subplot(1, 3, 3)## Visualizar la matriz de valor (V)
plt.imshow(v, cmap='gray', vmin=0, vmax=255)  # El valor tiene un rango de 0 a 255 en OpenCV
plt.title('Value')
plt.colorbar()

plt.tight_layout()
plt.show()

## Definir la máscara HSV para un rango de tonos verdes
lower_green = np.array([36,0,17])
upper_green = np.array([55,255,200])
mask = cv2.inRange(hsv_image, lower_green, upper_green)

## Aplicar la máscara a los valores de tono, saturación y valor
h_masked = h[mask > 0]
s_masked = s[mask > 0]
v_masked = v[mask > 0]

plt.figure(figsize=(12, 6)) ## Crear un boxplot para cada componente

## Boxplot para el tono (H)
plt.subplot(1, 3, 1)
plt.boxplot(h.flatten(), vert=False)
plt.scatter(h_masked, np.full_like(h_masked, 1), alpha=0.5, color='green')
plt.title('Hue')
plt.xlabel('Hue Value')

## Boxplot para la saturación (S)
plt.subplot(1, 3, 2)
plt.boxplot(s.flatten(), vert=False)
plt.scatter(s_masked, np.full_like(s_masked, 1), alpha=0.5, color='green')
plt.title('Saturation')
plt.xlabel('Saturation Value')

## Boxplot para el valor (V)
plt.subplot(1, 3, 3)
plt.boxplot(v.flatten(), vert=False)
plt.scatter(v_masked, np.full_like(v_masked, 1), alpha=0.5, color='green')
plt.title('Value')
plt.xlabel('Value')

plt.tight_layout()
plt.show()

def contar_arboles(imagen):
    img=cv2.imread(imagen) ##Cargar la imagen
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_green = np.array([36,0,17])
    upper_green = np.array([55,255,200])

    mask=cv2.inRange(hsv,lower_green,upper_green) ##Crear máscara

    green_img = cv2.bitwise_and(img,img,mask=mask) ##Aplicar máscara

    ## Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(green_img, cv2.COLOR_BGR2GRAY)
    ## Aplicar un umbral para obtener una imagen binaria
    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    ## Encontrar contornos en la imagen binaria}
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_arboles = len(contours)     ## Contar el número de contornos (árboles)
    cv2.imwrite("logo1.JPG", mask)
    return num_arboles

ruta_imagen = 'imagen_aerea2.JPG' ## Ruta de la imagen aérea

## Contar árboles en la imagen
numero_de_arboles = contar_arboles(ruta_imagen)
print("Número de árboles verdes en la imagen:", numero_de_arboles)