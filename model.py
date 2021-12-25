import tensorflow as tf

def encoding_block(inp, filters, dropout_prob=0.5):
	X = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l2')(inp)
	X = tf.keras.layers.BatchNormalization()(X)
	X = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l2')(X)
	X = tf.keras.layers.BatchNormalization()(X)
	X = tf.keras.layers.MaxPooling2D()(X)
	X = tf.keras.layers.Dropout(dropout_prob)(X)

	return X

def decoding_block(inp, res_con, filters, dropout_prob=0.5):
	X = tf.keras.layers.Conv2DTranspose(filters, 3, 2, activation='relu', padding='same', kernel_regularizer='l2')(inp)

	if res_con is None:
		merge = X
	else:
		merge = tf.concat((X, res_con), axis=-1)
	
	X = tf.keras.layers.Dropout(dropout_prob)(merge)
	X = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l2')(X)
	X = tf.keras.layers.BatchNormalization()(X)
	X = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer='l2')(X)
	X = tf.keras.layers.BatchNormalization()(X)

	return X

inp = tf.keras.layers.Input(shape=(None, None, 3))

conv1 = encoding_block(inp, 64)
conv2 = encoding_block(conv1, 128)
conv3 = encoding_block(conv2, 256)
conv4 = encoding_block(conv3, 512)
conv5 = encoding_block(conv4, 512)

conv6 = decoding_block(conv5, conv4, 512)
conv7 = decoding_block(conv6, conv3, 256)
conv8 = decoding_block(conv7, conv2, 128)
conv9 = decoding_block(conv8, conv1, 64)
conv10 = decoding_block(conv9, None, 64)

conv11 = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(conv10)

unet = tf.keras.Model(inputs=inp, outputs=conv11)

lr = 0.001
unet.compile(
	optimizer=tf.keras.optimizers.Adam(lr), 
	loss=tf.keras.losses.BinaryCrossentropy(), 
	metrics=['accuracy']
)

unet.summary()

