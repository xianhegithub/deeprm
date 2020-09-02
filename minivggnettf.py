# import the necessary packages
import tensorflow as tf

class MiniVGGNetTF:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the input shape and channel dimension, assuming
		# TensorFlow/channels-last ordering
		inputShape = (height, width, depth)
		chanDim = -1

		# define the model input
		inputs = tf.keras.layers.Input(shape=inputShape)
		x = tf.keras.layers.Flatten()(inputs)
		l_hid = tf.keras.layers.Dense(20,
									  activation='relu',
									  kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
									  bias_initializer='zeros'
									  )(x)

		#    l_out = lasagne.layers.DenseLayer(l_hid, num_units=output_length, nonlinearity=lasagne.nonlinearities.softmax,
		#        W=lasagne.init.Normal(.01), b=lasagne.init.Constant(0) )
		l_out = tf.keras.layers.Dense(classes,
									  activation='softmax',
									  kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
									  bias_initializer='zeros'
									  )(l_hid)

		# create the model
		model = tf.keras.models.Model(inputs, l_out, name="minivggnet_tf")

		return model
