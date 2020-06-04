import tensorflow as tf
import numpy as np

def build_deep_autoencoder(img_shape, code_size):
    """
    Makes a Deep Autoencoder (Encoder-Decoder pair)

    :param img_shape: Shape of the image that is being encoded
    :param code_size: Number of the neurons in the bottleneck layer
    :returns: The full autoencoder model compiled with MSE loss.
    """
    encoder = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(img_shape),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='elu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(code_size),
    ], name='Encoder')
    decoder = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((code_size,)),
        tf.keras.layers.Dense((img_shape[0] // 4) * (img_shape[1] // 4) * 32),
        tf.keras.layers.Reshape((img_shape[0] // 4, img_shape[1] // 4, 32)),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'),
        tf.keras.layers.Conv2DTranspose(filters=img_shape[2], kernel_size=(3, 3), strides=2, activation='elu', padding='same'),
    ], name='Decoder')
    input_image = tf.keras.layers.Input(img_shape, name='Image_Input')
    code = encoder(input_image)
    output_image = decoder(code)
    autoencoder = tf.keras.models.Model(inputs=input_image, outputs=output_image)
    autoencoder.compile(optimizer="adamax", loss='mse')
    return autoencoder

model = build_deep_autoencoder(img_shape=(28, 28, 1), code_size=100)
model.summary()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
model.fit(x=x_train, y=x_train, validation_data=[x_test, x_test], epochs=5)