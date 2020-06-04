import tensorflow as tf
import numpy as np

def build_generative_adverserial_net(img_shape, code_size):
    """
    Makes a Deep Autoencoder (Encoder-Decoder pair)

    :param img_shape: Shape of the image that is being encoded
    :param code_size: Number of the neurons in the bottleneck layer
    :returns: The full autoencoder model compiled with MSE loss.
    """
    generator = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer([code_size],name='noise'),
        tf.keras.layers.Dense(10*8*8, activation='elu'),
        tf.keras.layers.Reshape((8,8,10)),
        tf.keras.layers.Deconv2D(64, kernel_size=(5,5),activation='elu'),
        tf.keras.layers.Deconv2D(64, kernel_size=(5,5),activation='elu'),
        tf.keras.layers.UpSampling2D(size=(2,2)),
        tf.keras.layers.Deconv2D(32, kernel_size=3,activation='elu'),
        tf.keras.layers.Deconv2D(32, kernel_size=3,activation='elu'),
        tf.keras.layers.Deconv2D(32, kernel_size=3,activation='elu'),
        tf.keras.layers.Conv2D(3, kernel_size=3,activation=None),
    ], name='Generator')

    discriminator = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(img_shape),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation='tanh'),
        tf.keras.layers.Dense(2,activation=tf.nn.log_softmax),
    ], name='Discriminator')

    input_code = tf.placeholder('float32' ,[None, code_size])
    real_data = tf.placeholder('float32', [None,] + list(img_shape))
    logp_real = discriminator(real_data)
    generated_data = generator(input_code)
    logp_gen = discriminator(generated_data)

    d_loss = -tf.reduce_mean(logp_real[:,1] + logp_gen[:,0])
    d_loss += tf.reduce_mean(discriminator.layers[-1].kernel**2)
    disc_optimizer =  tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss,var_list=discriminator.trainable_weights)
    g_loss = tf.reduce_mean(logp_gen[:,0])
    gen_optimizer = tf.train.AdamOptimizer(1e-4).minimize(g_loss,var_list=generator.trainable_weights)


model = build_generative_adverserial_net(img_shape=(28, 28, 1), code_size=256)
model.summary()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
model.fit(x=x_train, y=x_train, validation_data=[x_test, x_test], epochs=5)