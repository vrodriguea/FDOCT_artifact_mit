import numpy as np
from random import randint, random
from numpy import ones, zeros, asarray
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Conv2DTranspose, Activation, Concatenate
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt

def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=0.5)
    return model

def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Concatenate()([g, input_layer])
    return g

def define_generator(image_shape, n_resnet=2):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    for _ in range(n_resnet):
        g = resnet_block(256, g)

    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(1, (7, 7), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    model = Model(in_image, out_image)
    return model

def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    output_f = g_model_2(gen1_out)
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
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


# def load_real_samples(filename):
#     data = np.load(filename)
#     X1, X2 = data['train_dominio_1'], data['train_dominio_2']
#     X1 = (X1 - 127.5) / 127.5
#     X2 = (X2 - 127.5) / 127.5
#     return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

def generate_fake_samples(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def save_models(step, g_model_AtoB, g_model_BtoA):
    filename1 = 'g_model_AtoB_%06d.h5' % (step + 1)
    g_model_AtoB.save(filename1)
    filename2 = 'g_model_BtoA_%06d.h5' % (step + 1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

def summarize_performance(step, g_model, trainX, name, n_samples=5):
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_in[i])
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_out[i])
    filename1 = '%s_generated_cycle_plot_%06d.png' % (name, (step + 1))
    plt.savefig(filename1)
    plt.close()

def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0, len(pool) - 1)
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    n_epochs, n_batch = 80, 16
    n_patch = d_model_A.output_shape[1]

    trainA, trainB = dataset
    poolA, poolB = list(), list()
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    dA_losses, dB_losses, g_losses1, g_losses2 = [], [], [], []

    for i in range(n_steps):
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        g_loss2 = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realA, X_realB, X_realA])
        g_loss1 = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realB, X_realA, X_realB])

        print(f'>{i + 1}, dA[{dA_loss1:.3f},{dA_loss2:.3f}] dB[{dB_loss1:.3f},{dB_loss2:.3f}] g[{g_loss1[0]:.3f},{g_loss2[0]:.3f}]')

        dA_losses.append((dA_loss1 + dA_loss2) / 2)
        dB_losses.append((dB_loss1 + dB_loss2) / 2)
        g_losses1.append(g_loss1[0])
        g_losses2.append(g_loss2[0])

        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')

        if (i + 1) % (bat_per_epo * 10) == 0:
            save_models(i, g_model_AtoB, g_model_BtoA)

    # plt.figure(figsize=(10, 5))
    # plt.plot(dA_losses, label='Discriminator A Loss')
    # plt.plot(dB_losses, label='Discriminator B Loss')
    # plt.plot(g_losses1, label='Generator Loss BtoA')
    # plt.plot(g_losses2, label='Generator Loss AtoB')
    # plt.title("Losses over Epochs")
    # plt.xlabel("Steps")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

dataset = load_real_samples('dataset_cycle.npz')
image_shape = dataset[0].shape[1:]

d_model_A = define_discriminator(image_shape)
d_model_B = define_discriminator(image_shape)
g_model_AtoB = define_generator(image_shape)
g_model_BtoA = define_generator(image_shape)
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)

