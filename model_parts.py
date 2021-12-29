import tensorflow as tf

class DoubleConv():
	def __init__(self, out_ch):
		self.out_ch = out_ch 

	def __call__(self, x, dropout, kernel_regularizer):
		x = tf.keras.layers.Conv2D(filters=self.out_ch, kernel_size=3, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(x)
		x = tf.keras.layers.Dropout(dropout)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)
		x = tf.keras.layers.Conv2D(filters=self.out_ch, kernel_size=3, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(x)
		x = tf.keras.layers.Dropout(dropout)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Activation('relu')(x)

		return x

class DownSample():
	def __init__(self, out_ch):
		self.out_ch = out_ch 

	def __call__(self, x, dropout=0, kernel_regularizer=None):
		x = tf.keras.layers.MaxPooling2D()(x)
		x = DoubleConv(self.out_ch)(x, dropout, kernel_regularizer)
		
		return x

class UpSample():
	def __init__(self, out_ch):
		self.out_ch = out_ch 

	def __call__(self, x, y, dropout=0, kernel_regularizer=None):
		x = tf.keras.layers.Conv2DTranspose(self.out_ch, kernel_size=3, strides=2, padding='same')(x)
		if y is not None:
			x = tf.concat([x, y], axis=-1)
		x = DoubleConv(self.out_ch)(x, dropout, kernel_regularizer)

		return x