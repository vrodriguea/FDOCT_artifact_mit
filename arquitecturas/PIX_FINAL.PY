#%%%
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU
from numpy import asarray, load, ones, savez_compressed, vstack, zeros
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, MaxPooling2D
from keras.initializers import RandomNormal
from matplotlib import pyplot as pyplots
from keras.optimizers import Adam
from numpy.random import randint
import matplotlib.pyplot as plt
from keras.models import Model
from os.path import join
from os import listdir
import numpy as np
import re
import os


#%%%

def define_discriminator(image_shape):
	"""
    Define y compila un modelo discriminador para una GAN.

    Este modelo toma dos imágenes como entrada: una imagen fuente y una imagen objetivo (o su copia espejo),
    y aprende a distinguir entre combinaciones reales y falsas de estas imágenes.
    La salida es un mapa de características que indica las partes de la imagen que son reales o falsas.

    Parámetros:
    - image_shape: Una tupla que define la forma de las imágenes de entrada. Debe ser en el formato (altura, anchura, canales).

    Retorna:
    - model: Un modelo de Keras compilado que actúa como el discriminador en una GAN
	
    """

    # Inicialización de pesos
	init = RandomNormal(stddev=0.02)
	# imagen muestra con imagen espejo 
	in_src_image = Input(shape=image_shape)
	# imagen sin copia
	in_target_image = Input(shape=image_shape)
	
    # Concatenación de imágenes a lo largo del eje de los canales
	merged = Concatenate()([in_src_image, in_target_image])
	
    # Bloque de convolución C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	# Capa de salida,
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
    # Salida binaria que indica si las imágenes son reales o falsas
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
    # Definición del modelo
	model = Model([in_src_image, in_target_image], patch_out)
	
	# Compilación del modelo
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=0.5)
	return model


# Encoder 
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	"""
    Define un bloque codificador para una CNN.
    
    Parámetros:
    - layer_in: Capa de entrada para el bloque codificador.
    - n_filters: Número de filtros para la capa Conv2D.
    - batchnorm: Booleano, indica si incluir normalización por lotes.
    
    Retorna:
    - g: La capa de salida del bloque codificador.
    """
	init = RandomNormal(stddev=0.02)
	# Downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # Normalización por lotes
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# Activación Leaky ReLU
	g = LeakyReLU(alpha=0.2)(g)
	return g



# Decoder
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	"""
    Define un bloque decodificador para una cnn.
    
    Parámetros:
    - layer_in: Capa de entrada para el bloque decodificador.
    - skip_in: Capa para la conexión de salto.
    - n_filters: Número de filtros para la capa Conv2DTranspose.
    - dropout: Booleano, indica si incluir capa de abandono.
    
    Retorna:
    - g: La capa de salida del bloque decodificador.
    """
	init = RandomNormal(stddev=0.02)
	# Upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# Batch normalization
	g = BatchNormalization()(g, training=True)
	# Dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# Uso de skip connection
	g = Concatenate()([g, skip_in])
    # Activación ReLU
	g = Activation('relu')(g)
	return g

# Genrador
def define_generator(image_shape=(256,256,1)):
	"""
    Define el modelo generador para la red GAN.
    
    Parámetros:
    - image_shape: Forma de la imagen de entrada.
    
    Retorna:
    - model: El modelo generador compilado.
    """

	init = RandomNormal(stddev=0.02)
	# imagen de entrada
	in_image = Input(shape=image_shape)
	# Encoder
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	
	# Cuello de botella, sin normalización por lotes y activación ReLU
	b = Conv2D(512, (4,4), strides=(2,2), padding='same',
			 kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	
	# Decoder
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	
    # Imagen de salida
	out_image = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(out_image)# Activación tanh
	
	# Modelo
	model = Model(in_image, out_image)
	return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	"""
    Define y compila el modelo GAN combinando el generador y el discriminador.
    
    Parámetros:
    - g_model: El modelo generador.
    - d_model: El modelo discriminador.
    - image_shape: La forma de la imagen de entrada.
    
    Retorna:
    - model: El modelo GAN compilado.
    """
	# pesos del discriminador no se actualizan
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
			
	# Imagen de entrada
	in_src = Input(shape=image_shape)
	# imagen de entrada conectada al generador
	gen_out = g_model(in_src)
	# conectar la imagen de entrada y la salida del generador al discriminador
	dis_out = d_model([in_src, gen_out])
	
	# imagen de entrada, salida del generador y salida del discriminador
	model = Model(in_src, [dis_out, gen_out])
	
	# Model
	opt = Adam(learning_rate=0.0002, beta_1=0.5) 
	model.compile(loss='mae', optimizer=opt)
	return model


def load_real_samples(filename):
	"""
	Define y compila el modelo GAN combinando el generador y el discriminador.

	Parámetros:
	- filename: El nombre del archivo que contiene los datos de muestra reales.
	
	Retorna:
	- samples: Una lista que contiene las muestras reales escaladas.
	"""
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	
	# Escala de [0,255] a [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	samples = [X1, X2]
	
	return samples

def generate_real_samples(dataset, n_samples, patch_shape):
	"""
	Genera un lote de muestras reales a partir del conjunto de datos.
	
	Parámetros:
	- dataset: El conjunto de datos que contiene pares de imágenes (trainA, trainB).
	- n_samples: Número de muestras a generar.
	- patch_shape: La forma del parche para las etiquetas de clase.
	
	Retorna:
	- [X1, X2]: Un par de arrays con las imágenes seleccionadas de trainA y trainB.
	- y: Las etiquetas de clase 'reales' (1) para las muestras generadas.
	"""
	
	trainA, trainB = dataset
	# Elige instancias aleatorias
	ix = randint(0, trainA.shape[0], n_samples)
	# Recupera imágenes seleccionadas
	X1, X2 = trainA[ix], trainB[ix]
    # Genera etiquetas de clase reales - etiqueta 1
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	
	return [X1, X2], y



def generate_fake_samples(g_model, samples, patch_shape):
	
    """
    Genera un lote de imágenes falsas utilizando el modelo generador.
    
    Parámetros:
    - g_model: El modelo generador que se utilizará para generar las imágenes falsas.
    - samples: Las muestras de entrada para el modelo generador.
    - patch_shape: La forma del parche para las etiquetas de clase.
    
    Retorna:
    - X: Las imágenes falsas generadas por el modelo.
    - y: Las etiquetas de clase 'falsas' (0) para las imágenes generadas.
    """
    X = g_model.predict(samples)
    #  # Crea etiquetas de clase falsas - etiqueta 0
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, g_model, dataset, n_samples=3):
	"""
    Resumen del rendimiento del modelo generador.
    
    Parámetros:
    - step: Paso de entrenamiento actual.
    - g_model: Modelo generador.
    - dataset: Conjunto de datos que contiene pares de imágenes (realA, realB).
    - n_samples: Número de muestras para generar y visualizar.
    """
    # Selecciona una muestra de imágenes de entrada
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	## Genera un lote de muestras falsas
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# Escala de[-1,1] a [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# Grafica las imágenes de entrada, 
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.title('Real A' if i == 0 else '')  
		pyplot.imshow(X_realA[i])
	# Grafica la imagen objetivo generada
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.title('Fake B' if i == 0 else '')  
		pyplot.imshow(X_fakeB[i])
	# Grafica la imagen objetivo real
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.title('Real B' if i == 0 else '')
		pyplot.imshow(X_realB[i])
		
    # Guarda la gráfica en un archivo
	filename1 = 'plot_NORMA2_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_pix2pix_norma2_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))



def train(d_model, g_model, gan_model, dataset, n_epochs=5, n_batch=1):
	
    """
    Entrena modelos GAN: Discriminador (d_model), Generador (g_model) y el modelo GAN combinado (gan_model).
    
    Entradas:
    - d_model: Modelo del discriminador, que intenta distinguir entre imágenes reales y generadas.
    - g_model: Modelo del generador, que intenta generar imágenes que parezcan reales.
    - gan_model: Modelo combinado de GAN que actualiza el generador basándose en el desempeño del discriminador.
    - dataset: Tupla de datos (trainA, trainB), donde trainA son las imágenes de entrada y trainB son las imágenes objetivo.
    - n_epochs: Número de épocas para el entrenamiento.
    - n_batch: Tamaño del lote para el entrenamiento.
    
    Salidas:
    Esta función no retorna valores explícitamente, pero realiza varias acciones:
    - Entrena los modelos d_model, g_model y gan_model con los datos proporcionados.
    - Imprime el desempeño de los modelos en cada iteración y en puntos específicos del entrenamiento.
    - Guarda los modelos y genera gráficas de las pérdidas durante el entrenamiento.
    """
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # dataset
    trainA, trainB = dataset
	
    # Calcula el número de lotes por época de entrenamiento
    bat_per_epo = int(len(trainA) / n_batch)
    # Calcula el número total de iteraciones de entrenamiento
    d_loss_list, g_loss_list = [], []
    n_steps = bat_per_epo * n_epochs
    d_loss_accum, g_loss_accum = [], []

    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        results = gan_model.train_on_batch(X_realA, [y_real, X_realB])# summarize performance
        g_loss = results[0]
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)
        # Store losses
        d_loss_list.append((d_loss1 + d_loss2) / 2)
        g_loss_list.append(g_loss)
		
         # Al final de cada época, calcula el promedio de las pérdidas acumuladas y reinicia las listas
        if (i+1) % bat_per_epo == 0:
            epoch_d_loss = sum(d_loss_accum) / len(d_loss_accum)
            epoch_g_loss = sum(g_loss_accum) / len(g_loss_accum)
            d_loss_list.append(epoch_d_loss)
            g_loss_list.append(epoch_g_loss)
            d_loss_accum, g_loss_accum = [], []  # Reinicia las listas para la próxima época

                
    # cambiar por epocas 
    plt.plot(d_loss_list, label='Discriminator Loss')
    plt.plot(g_loss_list, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    
    plt.savefig('training_losses_pix_norm2.png')    
    plt.clf()


dataset = load_real_samples('dataset_intensidad.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)

train(d_model, g_model, gan_model, dataset)
#%%