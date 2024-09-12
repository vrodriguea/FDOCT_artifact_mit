
#%%
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU
from numpy import asarray, load, ones, savez_compressed, vstack, zeros
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, MaxPooling2D
from keras.initializers import RandomNormal
from matplotlib import pyplot as pyplots
from keras.optimizers import Adam, RMSprop, SGD
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


# def define_gan(g_model, d_model, image_shape):

# 	for layer in d_model.layers:
# 		if not isinstance(layer, BatchNormalization):
# 			layer.trainable = False
			
# 	in_src = Input(shape=image_shape)
# 	gen_out = g_model(in_src)
# 	dis_out = d_model([in_src, gen_out])
	
# 	model = Model(in_src, [dis_out, gen_out])
	
# 	opt = Adam(learning_rate=0.0002, beta_1=0.5) 
# 	model.compile(loss='mae', optimizer=opt)
# 	return model

def define_gan(g_model, d_model, image_shape, optimizer):

    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
            
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    
    model = Model(in_src, [dis_out, gen_out])
    model.compile(loss='mae', optimizer=optimizer)  # Usa el optimizador que se pase como argumento
    return model


def normalizar(tomo):

    tomo_min = np.min(tomo)
    tomo_max = np.max(tomo)
    
    normalizaso = (tomo - tomo_min) / (tomo_max - tomo_min)
    
    return normalizaso, tomo_min, tomo_max
    
def load_real_samples(filename):
    data = load(filename)
    
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
		
	filename1 = 'plot_exp_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	filename2 = 'model_pix2pix_exp_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))



def train(d_model, g_model, gan_model, dataset, optimizer_name='adam', n_epochs=5, n_batch=1):
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
            print(f'Epoch {current_epoch}, Optimizer: {optimizer_name}, d_loss: {d_loss_epoch[-1]:.3f}, g_loss: {g_loss_epoch[-1]:.3f}')
            
            if current_epoch % 2 == 0:
                summarize_performance(current_epoch * bat_per_epo, g_model, dataset)
    
    plt.plot(d_loss_epoch, label=f'Discriminator Loss ({optimizer_name})')
    plt.plot(g_loss_epoch, label=f'Generator Loss ({optimizer_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Losses by Epoch - {optimizer_name}')
    plt.legend()
    plt.savefig(f'training_losses_by_epoch_{optimizer_name}.png')
    plt.clf()


#%%


optimizers = {
    'adam': Adam(learning_rate=0.0002, beta_1=0.5),
    'rmsprop': RMSprop(learning_rate=0.0002),
    'sgd': SGD(learning_rate=0.0002, momentum=0.9)
}

dataset = load_real_samples('dataset_solap.npz')
image_shape = dataset[0].shape[1:]

for opt_name, optimizer in optimizers.items():
    print(f'\nTraining with {opt_name.upper()} optimizer...')
    
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape, optimizer)

    train(d_model, g_model, gan_model, dataset, optimizer_name=opt_name)


# %%
def compare_optimizers():
    optimizers = ['adam', 'rmsprop', 'sgd']
    
    for opt_name in optimizers:
        img = plt.imread(f'training_losses_by_epoch_{opt_name}.png')
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Training Losses - {opt_name.upper()}')
        plt.show()

compare_optimizers()

