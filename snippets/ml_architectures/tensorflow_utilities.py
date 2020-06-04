from collections import defaultdict
import numpy as np
import tensorflow as tf


class ModelSaveCallback(keras.callbacks.Callback):
    """
    Provides a Callback to store the weights of a model in an intermediate
    epoch using a callback.
    """

    def __init__(self, file_name):
        super(ModelSaveCallback, self).__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        model_filename = self.file_name.format(epoch)
        tf.keras.models.save_model(self.model, model_filename)
        print("Model saved in {}".format(model_filename))


def reset_tf_session():
    """
    Resets Tensorflow session so that we can start
    building the computation graph from scratch.
    """
    curr_session = tf.get_default_session()
    if curr_session is not None:
        curr_session.close()
    tf.keras.backend.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    tf.keras.backend.set_session(s)
    return s
