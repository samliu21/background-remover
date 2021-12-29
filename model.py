import tensorflow as tf

from model_parts import DownSample, UpSample

class Unet():
	def __init__(self):
		self.inp = tf.keras.layers.Input(shape=(None, None, 3))

		self.conv1 = DownSample(64)(self.inp)
		self.conv2 = DownSample(128)(self.conv1)
		self.conv3 = DownSample(256)(self.conv2)
		self.conv4 = DownSample(512)(self.conv3)
		self.conv5 = DownSample(512)(self.conv4, dropout=0.4, kernel_regularizer='l2')

		self.conv6 = UpSample(512)(self.conv5, self.conv4)
		self.conv7 = UpSample(256)(self.conv6, self.conv3)
		self.conv8 = UpSample(128)(self.conv7, self.conv2)
		self.conv9 = UpSample(64)(self.conv8, self.conv1)
		self.conv10 = UpSample(64)(self.conv9, None, dropout=0.4, kernel_regularizer='l2')

		self.conv11 = tf.keras.layers.Conv2D(1, 1, padding='same')(self.conv10)
		self.conv11 = tf.keras.layers.Activation('sigmoid')(self.conv11)

		self.model = tf.keras.Model(inputs=self.inp, outputs=self.conv11)

		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			0.002,
			decay_steps=300,
			decay_rate=0.98,
			staircase=True,
		)

		self.model.compile(
			optimizer=tf.keras.optimizers.Adam(lr_schedule), 
			loss=tf.keras.losses.BinaryCrossentropy(),
			metrics=['accuracy', IoU],
		)

	def get_model(self):
		return self.model

def IoU(y_true, y_pred):
	intersection = tf.reduce_sum(y_true * y_pred)
	y_pred = tf.cast(y_pred >= 0.5, tf.float32)
	union = tf.reduce_sum(tf.cast((y_true + y_pred) >= 1, tf.float32)) + 1e-6
	return intersection / union

unet = Unet().get_model()

unet.summary()