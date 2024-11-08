import os
import re
from os import listdir
from os.path import join

import numpy as np
from numpy import asarray, load, ones, savez_compressed, vstack, zeros
from numpy.random import randint

import matplotlib.pyplot as plt
from matplotlib import pyplot

from keras.initializers import RandomNormal
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

import tensorflow as tf
from tensorflow.keras.activations import gelu
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (Input, MaxPooling2D, SeparableConv2D, UpSampling2D, Multiply, concatenate)
from tensorflow.keras.models import Model


def define_discriminator(image_shape):


	init = RandomNormal(stddev=0.02)
	in_src_image = Input(shape=image_shape)
	in_target_image = Input(shape=image_shape)
	
	merged = Concatenate()([in_src_image, in_target_image])
	
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
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
	patch_out = Activation('sigmoid')(d)

	model = Model([in_src_image, in_target_image], patch_out)
	
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=0.5)
	return model



def unet_generator(input_shape):
    inputs = Input(input_shape)

    def encoder_block(x, filters):
        x = SeparableConv2D(filters, 7, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        p = SeparableConv2D(filters, 2, strides=2, padding='same')(x)  
        p = Activation('gelu')(p)
        return x, p

    conv1, pool1 = encoder_block(inputs, 64)   # 512x512 256x256
    conv2, pool2 = encoder_block(pool1, 128)   # 256x256 128x128
    conv3, pool3 = encoder_block(pool2, 256)   # 128x128 64x64
    conv4, pool4 = encoder_block(pool3, 512)   # 64x64 32x32
    conv5, pool5 = encoder_block(pool4, 512)   # 32x32 16x16
    conv6, pool6 = encoder_block(pool5, 1024)  # 16x16 8x8
    conv7, pool7 = encoder_block(pool6, 1024)  # 8x8 4x4
    conv8, pool8 = encoder_block(pool7, 1024)  # 4x4 2x2
    conv9, pool9 = encoder_block(pool8, 1024)  # 2x2 1x1

    conv_bottleneck = SeparableConv2D(1024, 7, padding='same')(pool9)
    conv_bottleneck = Activation('gelu')(conv_bottleneck)

    def decoder_block(x, skip, filters):
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = Multiply()([skip, x])  #
        x = SeparableConv2D(filters, 7, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        return x
 
    conv10 = decoder_block(conv_bottleneck, conv9, 1024)  # 1x1  2x2
    conv11 = decoder_block(conv10, conv8, 1024)  # 2x2  4x4
    conv12 = decoder_block(conv11, conv7, 1024)  # 4x4  8x8
    conv13 = decoder_block(conv12, conv6, 512)   # 8x8  16x16
    conv14 = decoder_block(conv13, conv5, 512)   # 16x16  32x32
    conv15 = decoder_block(conv14, conv4, 512)   # 32x32  64x64
    conv16 = decoder_block(conv15, conv3, 256)   # 64x64  128x128
    conv17 = decoder_block(conv16, conv2, 128)   # 128x128  256x256
    conv18 = decoder_block(conv17, conv1, 64)    # 256x256  512x512

    decoded = SeparableConv2D(1, 3, activation='tanh', padding='same')(conv18)

    autoencoder = Model(inputs, decoded)
    return autoencoder


 
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


def total_loss(y_true, y_pred, L=0.05, M=0.15, N=0.8):
    content = content_loss(y_true, y_pred)
    feature = feature_loss(y_true, y_pred)
    adversarial = adversarial_loss(y_true, y_pred)
    total = L * content + M * feature + N * adversarial
    return total

def define_gan(g_model, d_model, image_shape, L=0.05, M=0.15, N=0.8):

    for layer in d_model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=lambda y_true, y_pred: total_loss(y_true, y_pred, L, M, N), optimizer=opt)
    
    return model


def normalizar(tomo):

    tomo_min = np.min(tomo)
    tomo_max = np.max(tomo)
    
    normalizaso = (tomo - tomo_min) / (tomo_max - tomo_min)
    
    return normalizaso, tomo_min, tomo_max
    
def load_real_samples(filename):
    data = np.load(filename)
    
    x1 = data['inputs_amplitud']
    y1 = data['targets_amplitud']
    
    x1_norma, x1_min, x1_max = normalizar(x1)
    y1_norma, y1_min, y1_max = normalizar(y1)
    
    return x1_norma, y1_norma


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

	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.title('Real A' if i == 0 else '')  
		pyplot.imshow(X_realA[i])
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.title('Fake B' if i == 0 else '')  
		pyplot.imshow(X_fakeB[i])
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.title('Real B' if i == 0 else '')
		pyplot.imshow(X_realB[i])
		
	filename1 = 'plot_Ccar_exp_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	filename2 = 'model_Ccar_exp_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


#%%

def train(d_model, g_model, gan_model, dataset, n_epochs=5, n_batch=1):
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
            
            if current_epoch % 2 == 0:
                summarize_performance(current_epoch * bat_per_epo, g_model, dataset)
    
    plt.plot(d_loss_epoch, label='Discriminator Loss')
    plt.plot(g_loss_epoch, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses by Epoch')
    plt.legend()
    plt.savefig('training_Ccar_exp.png') 
    plt.clf()
#%%

dataset = load_real_samples('dataset_solap.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
d_model = define_discriminator(image_shape)
g_model = unet_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)

train(d_model, g_model, gan_model, dataset)



