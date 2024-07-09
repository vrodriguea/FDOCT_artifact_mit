import tensorflow as tf
import numpy as np
import math
import os
#import cv2
import PIL
import numpy as np 
import os
from os.path import join
import matplotlib.pyplot as plt 
import re
import matplotlib.pyplot as plt
from skimage.io import imread
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint
import numpy as np 
import os
from os.path import join
import matplotlib.pyplot as plt 
import re
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
from tensorflow.keras import Model 

path = r'/home/vrodriguea/BD_SIM/simulados/tom_artifact/'
file_list = os.listdir(path)

tom1_real_files = sorted([file for file in file_list if file.startswith('tom1_real')])
tom1_imag_files = sorted([file for file in file_list if file.startswith('tom1_imag')])

tom1_data = {}

for filename_tom1_real, filename_tom1_imag in zip(tom1_real_files, tom1_imag_files):
    z, x, y = [int(re.search(f'{dim}=([0-9]+)', filename_tom1_real).group(1)) for dim in ['z', 'x', 'y']]

    tom1_real = np.fromfile(join(path, filename_tom1_real), 'single').reshape((z, x, y), order='F')
    tom1_imag = np.fromfile(join(path, filename_tom1_imag), 'single').reshape((z, x, y), order='F')

    tom1 = tom1_real + 1j * tom1_imag
    del tom1_real, tom1_imag
    filename_tom1 = filename_tom1_real.replace('_real', '')
    
    tom1_data[filename_tom1] = tom1

solapValues = []
solap = 70
maxSolap = z * 0.6

while solap <= maxSolap:
    solapValues.append(solap)
    solap = int(solap * 3.4)

solapValues = [int(value) for value in solapValues]

inputs_original = []
targets_original = []
inputs_ampli = []	
targets_ampli = []

for filename, tom1 in tom1_data.items():
    for solap in solapValues:
        z, x, y = tom1.shape  # Obtiene la forma de tom1
        newTom = np.zeros((2*z,x,y),dtype=complex)
        newTom2 = np.zeros((2*z,x,y),dtype=complex)

        newTom[solap:512+solap,:,:] = tom1 #solapado 
        newTom2[solap:512+solap,:,:] = tom1 #desplazdo 

        newTom[512-solap:1024-solap,:,:] = newTom[512-solap:1024-solap,:,:] + np.flip(tom1,axis=0)
        newTom2[512-solap:1024-solap,:,:] = newTom2[512-solap:1024-solap,:,:]

        center = newTom2.shape[0] // 2
        newTom2_cut = newTom2[center-256:center+256,:,:]

        center = newTom.shape[0] // 2
        newTom_cut = newTom[center-256:center+256,:,:]

        const = 0.0000001  # Una pequeña constante

        for i in range(newTom_cut.shape[2]):
            inputs_original.append(newTom_cut[:,:,i])
            inputs_original.append(newTom2_cut[:,:,i])

            transformed_input = 20 * np.log10(np.abs(newTom_cut[:,:,i]) + const)
            transformed_target = 20 * np.log10(np.abs(newTom2_cut[:,:,i]) + const)
            inputs_ampli.append(transformed_input)
            targets_ampli.append(transformed_target) 

inputs = np.expand_dims(np.array(inputs_ampli, dtype='float32'), -1)
targets = np.expand_dims(np.array(targets_ampli, dtype='float32'), -1)            
# def normalize(array):
#     return (array - array.min()) / (array.max() - array.min())

# inputs_s = normalize(inputs)
# targets_s = normalize(targets)

# def normalize(array):
#     return array / 255.0

# inputs_c = normalize(inputs)


def build_model(input_layer, start_neurons):
    # Start set of U-NET
    #dimesiones de entrada de 512x512, definidas en input layer 
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1) #tam de la ventana 
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle set of U-NET
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    # Final set of U-NET
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    #Here we have our output, and we made a resize for the compatibily with de model
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1) / 255

    return output_layer

class SequenceDataset(Sequence):

    def __init__(self, x_set, y_set, batch_size=32):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return (np.array([np.expand_dims(x, axis=2) for x in batch_x]),
                np.array([np.expand_dims(y, axis=2) for y in batch_y]))
        
        
#carga de datos

batch_size = 2
epochs = 100
x_set = inputs
y_set = targets

sequence = SequenceDataset(x_set, y_set, batch_size=batch_size)
input_layer = Input((512, 512, 1))
output_layer = build_model(input_layer, 16) #numero inicial de filtros 
unet = Model(input_layer, output_layer)

#modelo
unet.compile(optimizer="adam", loss="mse", metrics = ["mse"])
#entrenamiento 
history = unet.fit(x=x_set,y=y_set, batch_size= batch_size, epochs=epochs)
# unet.save(f'models/v1_unet.h5')
# plt.figure(figsize=(12, 8))
# plt.plot(history.history['loss'])
# plt.savefig('models/unet_loss_nonorm_v2.png')
# plt.close()


unet.save(f'models/v1_unet.h5')

plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'])
plt.title('Model Loss')  # Título del gráfico
plt.ylabel('Loss')  # Etiqueta del eje y
plt.xlabel('Epoch')  # Etiqueta del eje x
plt.savefig('models/unet_loss_nonorm_v2.png')
plt.close()
# %%
