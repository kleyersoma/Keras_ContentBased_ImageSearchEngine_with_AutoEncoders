# Import necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

class ConvAutoEncoder:
	
	# In this method by using the Functional API of Keras, we build an AutoEncoder Model
	@staticmethod
	def autoencoder_model(width, height, depth, filters=(32, 64), latentDim=16):
		"""
			Initialize the input shape to be "channels last" along with
			the channels dimension itselfchannels dimension itself
		""" 
		inputShape = (height, width, depth)
		chanDim = -1

		# Define the input to the Encoder
		inputs = Input(shape=inputShape)
		x = inputs

		# Loop over the number of filters
		for f in filters:
			# Apply a CONV -> RELU -> BN operation
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.1)(x)
			#x = BatchNormalization(axis=chanDim)(x)

		# Flatten the network and then construct the Latent vector
		volumeSize = K.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim, name="Encoded")(x)

		# Start building the Decoder model which will accept the output of the Encoder as its inputs
		x = Dense(np.prod(volumeSize[1:]))(latent)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

		# Loop over our number of filters again, but this time in reverse order
		for f in filters[::-1]:
			# Apply a CONV_TRANSPOSE -> RELU -> BN operation
			x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.1)(x)
			#x = BatchNormalization(axis=chanDim)(x)

		# Apply a single CONV_TRANSPOSE layer used to recover the original depth of the image
		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid", name="Decoded")(x)

		# Construct the AutoEncoder Model
		autoencoder = Model(inputs, outputs, name="AutoEncoder")

		# Return the Autoencoder Model
		return autoencoder