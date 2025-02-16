# redefine the custom functions used in the model

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom')
#  sample_z function
def sample_z(args):
    mu, sigma = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.0)
    return mu + K.exp(0.5 * sigma) * epsilon

@tf.keras.utils.register_keras_serializable(package='Custom')
class Sampling(Layer):
  def call(self, inputs):
      mu, sigma = inputs
      epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.0)
      return mu + K.exp(0.5 * sigma) * epsilon
  
  ## BCE

from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf
from tensorflow.keras import backend as K

@register_keras_serializable()
class VAELossLayer(Layer):
    def __init__(self, beta=1.0, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        original, reconstructed, mu, sigma = inputs

        original = tf.reshape(original, [tf.shape(original)[0], -1])
        reconstructed = tf.reshape(reconstructed, [tf.shape(reconstructed)[0], -1])

        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(original, reconstructed)
        )

        kl_loss = -0.5 * K.sum(1 + sigma - K.square(mu) - K.exp(sigma), axis=-1)
        kl_loss = self.beta * kl_loss
        total_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)

        return reconstructed
    
from tensorflow.keras.models import load_model


# Load the saved model
model = load_model("/Users/kristof/data/convVAE/ml_ops/model/vae_model_10_epochs_beta_7.h5",
                    custom_objects={'sample_z': sample_z,'Sampling': Sampling,'VAELossLayer': VAELossLayer})