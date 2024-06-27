#%%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, UpSampling2D, Multiply
from os import listdir
from os.path import join

import numpy as np
import re
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from matplotlib import pyplot
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Input
from keras.initializers import RandomNormal
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.models import Model
from keras.optimizers import Adam
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate 
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, Activation, Conv2DTranspose, Multiply
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose, Multiply
from tensorflow.keras.models import Model


def unet_generator(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = SeparableConv2D(64, 7, padding='same')(inputs)
    conv1 = Activation('gelu')(conv1)
    pool1 = SeparableConv2D(64, 2, strides=2, padding='same')(conv1)
    pool1 = Activation('gelu')(pool1)

    conv2 = SeparableConv2D(128, 7, padding='same')(pool1)
    conv2 = Activation('gelu')(conv2)
    pool2 = SeparableConv2D(128, 2, strides=2, padding='same')(conv2)
    pool2 = Activation('gelu')(pool2)

    conv3 = SeparableConv2D(256, 7, padding='same')(pool2)
    conv3 = Activation('gelu')(conv3)
    pool3 = SeparableConv2D(256, 2, strides=2, padding='same')(conv3)
    pool3 = Activation('gelu')(pool3)

    conv4 = SeparableConv2D(512, 7, padding='same')(pool3)
    conv4 = Activation('gelu')(conv4)
    pool4 = SeparableConv2D(512, 2, strides=2, padding='same')(conv4)
    pool4 = Activation('gelu')(pool4)

    conv5 = SeparableConv2D(512, 7, padding='same')(pool4)
    conv5 = Activation('gelu')(conv5)
    pool5 = SeparableConv2D(512, 2, strides=2, padding='same')(conv5)
    pool5 = Activation('gelu')(pool5)

    conv6 = SeparableConv2D(1024, 7, padding='same')(pool5)
    conv6 = Activation('gelu')(conv6)

    # Decoder
    up = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6)
    gate = Multiply()([conv5, up])
    conv7 = SeparableConv2D(512, 7, padding='same')(gate)
    conv7 = Activation('gelu')(conv7)

    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv7)
    gate1 = Multiply()([conv4, up1])
    conv8 = SeparableConv2D(512, 7, padding='same')(gate1)
    conv8 = Activation('gelu')(conv8)

    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv8)
    gate2 = Multiply()([conv3, up2])
    conv9 = SeparableConv2D(256, 7, padding='same')(gate2)
    conv9 = Activation('gelu')(conv9)

    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv9)
    gate3 = Multiply()([conv2, up3])
    conv10 = SeparableConv2D(128, 7, padding='same')(gate3)
    conv10 = Activation('gelu')(conv10)

    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv10)
    gate4 = Multiply()([conv1, up4])
    conv11 = SeparableConv2D(64, 7, padding='same')(gate4)
    conv11 = Activation('gelu')(conv11)

    decoded = SeparableConv2D(1, 3, activation='tanh', padding='same')(conv11)
    autoencoder = Model(inputs, decoded)

    return autoencoder


#%%
def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02) #desv stad inicializacion pesos capas 
    # imagen muestra con imagen espejo 
    in_src_image = Input(shape=image_shape) 
    # imagen sin copia
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image]) #imag en 1 tensor 
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d) #f activacion incluye negativos 

    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d) #normalización 
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)


    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d) #real o falsa (0 1)
 
    model = Model([in_src_image, in_target_image], patch_out) #tiene las dos img de entrada y salida 1-0

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=0.5) # minimizar la perdida 
    return model



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
    
    model.compile(loss=['binary_crossentropy', gan_loss], optimizer=opt, loss_weights=[1, 100])
    return model



def load_real_samples(filename):
 data = load(filename)
 X1, X2 = data['arr_0'], data['arr_1']
 X1 = (X1 - 127.5) / 127.5
 X2 = (X2 - 127.5) / 127.5
 return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
 trainA, trainB = dataset
 ix = randint(0, trainA.shape[0], n_samples)
 X1, X2 = trainA[ix], trainB[ix]
 y = np.ones((n_samples, patch_shape, patch_shape, 1))
 return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
 X = g_model.predict(samples)
 y = np.zeros((len(X), patch_shape, patch_shape, 1))
 return X, y

def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_ccar_norma_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))



def train(d_model, g_model, gan_model, dataset, n_epochs=1, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    d_loss_list, g_loss_list = [], [] 
    n_steps = bat_per_epo * n_epochs
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
        g_loss, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])# summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)
        # Store losses
        d_loss_list.append((d_loss1 + d_loss2) / 2)
        g_loss_list.append(g_loss)
    
    # Plot losses
    plt.plot(d_loss_list, label='Discriminator Loss')
    plt.plot(g_loss_list, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    
    plt.savefig('training_losses_ccar_norm.png')    
    plt.clf()


dataset = load_real_samples('dataset_intensidad.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = unet_generator(image_shape)
#gan_model = define_gan(g_model, d_model, image_shape)
gan_model = define_gan(g_model, d_model, image_shape, L=1.0, M=1.0, N=1.0)

train(d_model, g_model, gan_model, dataset)
# Train the models and save the history
#train(d_model, g_model, gan_model, dataset)


print("Discriminator Model Summary:")
d_model.summary()

print("\nGenerator Model Summary:")
g_model.summary()

print("\nGAN Model Summary:")
gan_model.summary()







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

from tensorflow.keras.applications import VGG19

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

def total_loss(y_true, y_pred, generator, discriminator, L=1.0, M=1.0, N=1.0):
    content = content_loss(y_true, y_pred)
    
    # Calcular la pérdida de características
    feature = feature_loss(y_true, y_pred)
    
    # Calcular la pérdida adversarial
    adversarial = adversarial_loss(y_true, y_pred)
    
    # Pérdida total
    total = L * content + M * feature + N * adversarial
    return total




# %%
# 
# import tensorflow as tf
# from tensorflow.keras import backend as K

# def contenido(y_true, y_pred):

#   mae = K.mean(K.abs(y_true - y_pred))

#   target_value = 5.0  

#   prediction_difference = K.mean(K.abs(y_pred - target_value))

#   penalty = prediction_difference * 0.1  
#   total_loss = mae + penalty

#   return total_loss




# def gram_matrix(input_tensor):
#     result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
#     input_shape = tf.shape(input_tensor)
#     num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
#     return result/(num_locations)


# def feature_loss(features, targets, weights=None):
#     if weights is None:
#         weights = [1.0/len(features)] * len(features)
        
#     gram_loss = 0
#     for f, t, w in zip(features, targets, weights):
#         gram_loss += tf.reduce_mean(tf.keras.losses.MeanSquaredError()(gram_matrix(f), gram_matrix(t))) * w
#     return gram_loss



#adversial loss def extract_features(model, x, layers):
    # outputs = [model.layers[i].output for i in layers]
    # model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    # return model(x)

def calc_Content_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1.0 / len(features)] * len(features)
    
    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += tf.reduce_mean(tf.square(f - t)) * w

    return content_loss

def gram(x):
    shape = tf.shape(x)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    features = tf.reshape(x, [b, h*w, c])
    gram_matrix = tf.matmul(features, features, transpose_a=True) / tf.cast(h*w, tf.float32)
    return gram_matrix

def calc_Gram_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1.0 / len(features)] * len(features)
        
    gram_loss = 0
    for f, t, w in zip(features, targets, weights):
        gram_loss += tf.reduce_mean(tf.square(gram(f) - gram(t))) * w
    return gram_loss

def calc_TV_Loss(x):
    tv_loss = tf.reduce_mean(tf.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + tf.reduce_mean(tf.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

def feature_loss(y_true, y_pred, model, layers, content_weights=None, gram_weights=None):
    true_features = extract_features(model, y_true, layers)
    pred_features = extract_features(model, y_pred, layers)
    
    content_loss = calc_Content_Loss(pred_features, true_features, weights=content_weights)
    
    gram_loss = calc_Gram_Loss(pred_features, true_features, weights=gram_weights)
    
    # Combina las pérdidas de contenido y Gram
    total_feature_loss = content_loss + gram_loss
    
    return total_feature_loss



#%%

# def total_loss(y_true, y_pred, l, m, n):
#     content = content_loss(y_true, y_pred)
#     feature = feature_loss(y_true, y_pred)
#     adversarial = adversarial_loss(y_true, y_pred)
    
#     loss = l * content + m * feature + n * adversarial
    
#     return loss






# # #%%
# import matplotlib.pyplot as plt


# # d_model = define_discriminator(input_shape)
# # g_model = unet_generator(input_shape)
# # gan_model = define_gan(g_model, d_model, input_shape)
# #%%
# import tensorflow as tf

# def custom_loss(y_true, y_pred):
#     # Calcula la diferencia cuadrada entre y_true y y_pred
#     squared_difference = tf.square(y_true - y_pred)
#     # Calcula la media de la diferencia cuadrada
#     mean_squared_difference = tf.reduce_mean(squared_difference)
#     return mean_squared_difference




# def create_cgan(generator, discriminator):
#     discriminator.trainable = False


#     label = Input(shape=(1,))


#     # Conectar la etiqueta como entrada al generador y al discriminador
#     z = Input(shape=(100,))
#     img = generator([z, label])
#     validity = discriminator([img, label])


#     combined = Model([z, label], validity)


#     return combined


# %%
# # Crear el generador y el discriminador
# input_shape = (512, 512, 1)
# # %%
# generator = unet_generator(input_shape)
# # %%
# discriminator = define_discriminator()
# # %%
# # Print generator model
# print("Generator Model:")
# g_model.summary()


# # Print discriminator model
# print("Discriminator Model:")
# d_model.summary()


# # CCAR
# #cgan = create_cgan(generator, discriminator)


# # Print CGAN model
# print("CGAN Model:")
# gan_model.summary()


# %%
#funciones de perdida 

# def content_loss(y_true, y_pred):
#     # Obtén las dimensiones de las imágenes
#     w, h = tf.shape(y_true)[1], tf.shape(y_true)[2]
    
#     diff = y_true - y_pred
#     loss = tf.reduce_sum(((diff + 1) ** 2 + 1) * diff) / (w * h)
    
#     return loss




# def total_loss(y_true, y_pred, l, m, n):
#     # Extrae los mapas de características de y_true y y_pred
#     vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
#     y_true_features = vgg(y_true)
#     y_pred_features = vgg(y_pred)
    
#     # Calcula las tres componentes de la pérdida
#     content = content_loss(y_true, y_pred)
#     feature = feature_loss(y_true_features, y_pred_features)
#     adversarial = adversarial_loss(y_true, y_pred)
    
#     # Combina las componentes de la pérdida con los pesos correspondientes
#     loss = l * content + m * feature + n * adversarial
    
#     return loss

# %%
# Suponiendo que 'real_output' son las predicciones del discriminador para datos reales
# y 'fake_output' son las predicciones del discriminador para datos generados

# def discriminator_loss(real_output, fake_output):
#     real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
#     fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss

# def generator_loss(fake_output):
#     return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

# def adversarial_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred))
