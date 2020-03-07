import tensorflow as tf
import tensorflow.keras.layers as L

def build_deep_autoencoder(img_shape, code_size):
    """
    Makes a Deep Autoencoder (Encoder-Decoder pair)

    :param img_shape: Shape of the image that is being encoded
    :param code_size: Number of the neurons in the bottleneck layer
    :returns: The full autoencoder model compiled with MSE loss.
    """
    
    encoder = tf.keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    encoder.add(L.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    encoder.add(L.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    encoder.add(L.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='valid'))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))

    decoder = tf.keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense((img_shape[0] // 16) * (img_shape[1] // 16) ** 256))
    decoder.add(L.Reshape((img_shape[0] // 16, img_shape[1] // 16, 256)))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    
    input_image = L.Input(img_shape)
    code = encoder(input_image)
    output_image = decoder(code)
    autoencoder = tf.keras.models.Model(inputs=input_image, outputs=output_image)
    autoencoder.compile(optimizer="adamax", loss='mse')

    return autoencoder

build_deep_autoencoder(img_shape=(32, 32, 3), code_size=100)