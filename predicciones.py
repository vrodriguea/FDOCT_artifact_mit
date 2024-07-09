#%%
from skimage.metrics import structural_similarity as ssim
import skimage.measure as skm
from keras.models import load_model
from numpy import savez_compressed
import matplotlib.pyplot as plt
from numpy import expand_dims
from os.path import join
import tensorflow as tf
from numpy import load
import numpy as np
import math
import cv2
import os
import re

#%%
def load_real_samples(filename):
 data = load(filename)
 x1 = data['inputs_amplitud']
 x1 = (x1 - 127.5) / 127.5
 y1 = data['targets_amplitud']
 y1 = (y1 - 127.5) / 127.5

 return x1,y1 
 

#%%
#hacerlo para cada set de val- predict- real 
dataset_vali = load_real_samples('dataset_inicial.npz')
# set_real =  load_real_samples('dataset_real.npz')
#%%
# muetsras originales
y_t = dataset_vali[0]
print(y_t.shape)
#%%
#para la prediccion, entrada con copia 
val_solap = dataset_vali[1]
print(val_solap.shape)

#%%
# Seleccionar la última muestra de val_solap
ultima_muestra = val_solap[-1:]  # Esto mantiene las dimensiones necesarias para la predicción

# Realizar la predicción con el modelo
prediccion = modelo_prueba.predict(ultima_muestra)

# Si deseas visualizar la forma de la predicción
print(prediccion.shape)

import matplotlib.pyplot as plt
import numpy as np

# Asumiendo que 'predictions' ya contiene las predicciones para 'val_solap'
# y que 'y_t' y 'val_solap' están definidos como se muestra anteriormente

# Índice de la última muestra
i = -1

plt.figure(figsize=(10, 6))

# Mostrar la última muestra de val_solap
plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(val_solap[i]), cmap='gray')
plt.title(f'Input Último')
plt.axis('off')

# Mostrar la predicción de la última muestra
plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(prediccion), cmap='gray')
plt.title(f'Predicción Último')
plt.axis('off')

# Mostrar el target de la última muestra
plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(y_t[i]), cmap='gray')
plt.title(f'Target Último')
plt.axis('off')

plt.tight_layout()
plt.show()




# #hacerlo para cada set de val- predict- real 
# dataset_vali = load_real_samples('dataset_validacion.npz')
# set_real =  load_real_samples('dataset_real.npz')
# # muetsras originales
# y_t = set_real[0]
# print(y_t.shape)
# #para la prediccion
# val_solap = dataset_vali[0]
# print(val_solap.shape)


from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from skimage import metrics as skm

# Lista de rutas a los archivos .h5 de los modelos
modelos_h5 = ['/content/drive/MyDrive/model_pix2pix_norma_395460.h5', 
              '/content/drive/MyDrive/model_pix2pix_norma_461370.h5',
              '/content/drive/MyDrive/model_pix2pix_norma_065910.h5']

# Asegurarse de que val_solap y y_t están definidos
# val_solap es tu conjunto de datos de entrada de validación
# y_t es tu conjunto de datos de objetivo de validación

# Seleccionar la última muestra de val_solap
ultima_muestra = val_solap[-1:]  # Esto mantiene las dimensiones necesarias para la predicción
target = y_t[-1:]

def psnr(original, reconstructed):
    """Calcula la Relación Señal-Ruido de Pico (PSNR) entre dos imágenes"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_value

def mae(original, reconstructed):
    """Calcula el Error Medio Absoluto (MAE) entre dos imágenes"""
    mae_value = np.mean(np.abs(original - reconstructed))
    return mae_value

def ssim(original, reconstructed):
    """Calcula el Índice de Similitud Estructural (SSIM) entre dos imágenes"""
    ssim_value = skm.structural_similarity(original, reconstructed, channel_axis=-1)
    return ssim_value

# DataFrame para almacenar las métricas
metricas_list = []

for modelo_path in modelos_h5:
    # Cargar el modelo
    modelo = load_model(modelo_path, compile=False)
    
    # Definir la función de predicción fuera del bucle
    @tf.function
    def predict(model, data):
        return model(data)

    # Realizar la predicción con el modelo cargado
    prediccion = predict(modelo, ultima_muestra).numpy()
    
    # Calcular métricas
    psnr_value = psnr(np.squeeze(target), np.squeeze(prediccion))
    mae_value = mae(np.squeeze(target), np.squeeze(prediccion))
    ssim_value = ssim(np.squeeze(target), np.squeeze(prediccion))
    
    # Agregar las métricas a la lista
    metricas_list.append({
        'Modelo': modelo_path.split('/')[-1],
        'PSNR': psnr_value,
        'MAE': mae_value,
        'SSIM': ssim_value
    })
    
    # Visualizar input, predicción y target
    plt.figure(figsize=(15, 5))
    
    # Mostrar la última muestra de val_solap (input)
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(ultima_muestra))
    plt.title(f'Input con {modelo_path}')
    plt.axis('off')
    
    # Mostrar la predicción
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(prediccion))
    plt.title(f'Predicción con {modelo_path}')
    plt.axis('off')
    
    # Mostrar el target de la última muestra
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(target))
    plt.title(f'Target con {modelo_path}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Convertir la lista de métricas en un DataFrame
metricas_df = pd.DataFrame(metricas_list)

# Mostrar el DataFrame con las métricas
print(metricas_df)


#%%
#leer modelo 
#lista de modelos


