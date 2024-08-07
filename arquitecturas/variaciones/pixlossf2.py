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


import matplotlib.pyplot as pyplot
from tensorflow.keras.applications import VGG19
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
#%%%

def define_discriminator(image_shape):

	init = RandomNormal(stddev=0.02)
	# imagen muestra con imagen espejo 
	in_src_image = Input(shape=image_shape)
	# imagen sin copia
	in_target_image = Input(shape=image_shape)
	
	merged = Concatenate()([in_src_image, in_target_image])
	
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
    # Salida binaria que indica si las  son reales o falsas
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	model = Model([in_src_image, in_target_image], patch_out)
	
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=0.5)
	return model


# Encoder 
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	"""
    Define un bloque codificador para una CNN.
    
    - layer_in: Capa de entrada para el bloque codificador.
    - n_filters: Número de filtros para la capa Conv2D.
    
    Retorna:
    - g: La capa de salida del bloque codificador.
    """
	init = RandomNormal(stddev=0.02)
	# Downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	g = LeakyReLU(alpha=0.2)(g)
	return g



# Decoder
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	"""
    Define un bloque decodificador para una cnn.
    
    - layer_in: Capa de entrada para el bloque decodificador.
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
	g = Activation('relu')(g)
	return g

# Genrador
def define_generator(image_shape=(256,256,1)):
	"""
    Define el modelo generador para la red GAN.
    
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
	out_image = Activation('tanh')(out_image)
	
	# Modelo
	model = Model(in_image, out_image)
	return model


def content_loss(y_true, y_pred):
    diff = y_true - y_pred
    diff_plus_one_squared = tf.square(diff + 1) + 1
    loss = tf.reduce_mean(diff_plus_one_squared * diff)
    return loss


def adversarial_loss(y_true, y_pred):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(y_pred), y_pred) * y_true
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(y_pred), y_pred) * (1 - y_true)
    total_loss = real_loss + fake_loss
    return total_loss

# red VGG19 preentrenada
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

def feature_loss(y_true, y_pred):
    if y_true.shape[-1] == 1:
        y_true = tf.image.grayscale_to_rgb(y_true)
        y_pred = tf.image.grayscale_to_rgb(y_pred)
    
    true_features = vgg(y_true)
    pred_features = vgg(y_pred)
    loss = tf.reduce_mean(tf.square(true_features - pred_features))
    return loss

def total_loss(y_true, y_pred, L=1.0, M=1.0, N=1.0):
    content = content_loss(y_true, y_pred)
    feature = feature_loss(y_true, y_pred)
    adversarial = adversarial_loss(y_true, y_pred)
    total = L * content + M * feature + N * adversarial
    return total


# define the combined generator and discriminator model, for updating the generator

def define_gan(g_model, d_model, image_shape, L=1.0, M=1.0, N=1.0):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    
    def gan_loss(y_true, y_pred):
        return total_loss(y_true, y_pred, L, M, N)
    
    model.compile(loss=[gan_loss], optimizer=opt)
    return model
#%%

def load_real_samples(filename):
	"""
	Define y compila el modelo GAN combinando el generador y el discriminador.

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
	
	:
	- dataset: El conjunto de datos que contiene pares de  (trainA, trainB).
	- n_samples: Número de muestras a generar.
	- patch_shape: La forma del parche para las etiquetas de clase.
	
	Retorna:
	- [X1, X2]: Un par de arrays con las  seleccionadas de trainA y trainB.
	- y: Las etiquetas de clase 'reales' (1) para las muestras generadas.
	"""
	
	trainA, trainB = dataset
	# Elige instancias aleatorias
	ix = randint(0, trainA.shape[0], n_samples)
	# Recupera  seleccionadas
	X1, X2 = trainA[ix], trainB[ix]
    # Genera etiquetas de clase reales - etiqueta 1
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	
	return [X1, X2], y



def generate_fake_samples(g_model, samples, patch_shape):
	
    """
    Genera un lote de  falsas utilizando el modelo generador.
    
    - g_model: El modelo generador que se  para generar las  falsas.
    - samples: Las muestras de entrada para el modelo generador.
    - patch_shape: La forma del parche para las etiquetas de clase.
    
    Retorna:
    - X: Las  falsas generadas por el modelo.
    - y: Las etiquetas de clase 'falsas' (0) para las  generadas.
    """
    X = g_model.predict(samples)
    #  # Crea etiquetas de clase falsas - etiqueta 0
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, g_model, dataset, n_samples=3):
	"""
    Resumen del rendimiento del modelo generador.
    
    - step: Paso de entrenamiento actual.
    - g_model: Modelo generador.
    - dataset: Conjunto de datos que contiene pares de  (realA, realB).
    - n_samples: Número de muestras para generar y visualizar.
    """
    # Selecciona una muestra de  de entrada
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	## Genera un lote de muestras falsas
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# Escala de[-1,1] a [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# Grafica las  de entrada, 
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
		
    # Guarda la  en un archivo
	filename1 = 'plot_NORMA2_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_pix2pix_norma2_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))




def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=16):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    d_loss_epoch, g_loss_epoch = [], []
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss_results = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        g_loss = g_loss_results[0]  

        if i % bat_per_epo == 0:
            d_losses, g_losses = [], []
        d_losses.append((d_loss1 + d_loss2) / 2)
        g_losses.append(g_loss)

        if (i + 1) % bat_per_epo == 0:
            d_loss_epoch.append(sum(d_losses) / len(d_losses))
            g_loss_epoch.append(sum(g_losses) / len(g_losses))
            current_epoch = (i + 1) // bat_per_epo
            print('Epoch %d, d_loss: %.3f, g_loss: %.3f' % (current_epoch, d_loss_epoch[-1], g_loss_epoch[-1]))
            
            if current_epoch % 10 == 0:
                summarize_performance(current_epoch * bat_per_epo, g_model, dataset)
    
    plt.plot(d_loss_epoch, label='Discriminator Loss')
    plt.plot(g_loss_epoch, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses by Epoch')
    plt.legend()
    plt.savefig('training_losses_by_epoch.png')
    plt.clf()

dataset = load_real_samples('dataset_intensidad.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)

train(d_model, g_model, gan_model, dataset)
#%%