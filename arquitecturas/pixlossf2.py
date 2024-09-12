from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU
from numpy import asarray, load, ones, savez_compressed, vstack, zeros
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, MaxPooling2D
from keras.initializers import RandomNormal
from matplotlib import pyplot as pyplots
from keras.optimizers import Adam
from numpy.random import randint
import matplotlib.pyplot as pyplot
from keras.models import Model
from os.path import join
from os import listdir
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot

from tensorflow.keras.applications import VGG19
import tensorflow as tf

#%%%

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


def define_encoder_block(layer_in, n_filters, batchnorm=True):

	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	g = LeakyReLU(alpha=0.2)(g)
	return g



def decoder_block(layer_in, skip_in, n_filters, dropout=True):

	init = RandomNormal(stddev=0.02)
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Concatenate()([g, skip_in])
	g = Activation('relu')(g)
	return g

def define_generator(image_shape=(256,256,1)):

	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=image_shape)
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
	
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	
	out_image = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(out_image)

	model = Model(in_image, out_image)
	return model


# cambio
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
		
	filename1 = 'plot_pixloss2_exp_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	filename2 = 'model_pixloss2_exp_%06d.h5' % (step+1)
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
    plt.savefig('training_pixloss_exp.png')
    plt.clf()


dataset = load_real_samples('data_combinado_final.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)

train(d_model, g_model, gan_model, dataset)