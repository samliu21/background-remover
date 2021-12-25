import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import unet

ds = tfds.load('oxford_iiit_pet:3.*.*')

train = ds['train']
test = ds['test']
all = train.concatenate(test)

DS_SIZE = len(all)
train_split = int(DS_SIZE * 0.9)
train = all.take(train_split)
test = all.skip(train_split)

LOAD_EXISTING_MODEL = True

def process_dataset(dataset, augment=False):
	"""
	Resize images to IMAGE_SIZE
	Convert segmentation mask to 0s and 1s for BinaryCrossentropy
	"""
	IMAGE_SIZE = (128, 128)

	dataset = dataset.map(
		lambda x: (
			tf.cast(tf.image.resize(x['image'], IMAGE_SIZE), tf.uint8),
			tf.cast(tf.image.resize(tf.cast(x['segmentation_mask'] != 2, tf.uint8), IMAGE_SIZE), tf.uint8),
		)
	).cache().shuffle(100).batch(32)

	def augment(img, mask):
		do_flip = tf.constant(np.random.rand() > 0.5)

		img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
		mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

		img = tf.image.random_brightness(img, 0.2)

		return (img, mask)
	
	if augment:
		dataset = dataset.map(lambda x, y: augment(x, y)).repeat(3)

	return dataset

def load_model():
	"""
	Load trained model
	"""
	reloaded = tf.keras.models.load_model('./unet.h5')

	return reloaded

train = process_dataset(train, augment=True)
test = process_dataset(test)

if LOAD_EXISTING_MODEL:
	model = load_model()
else:
	model = unet

history = model.fit(train, epochs=2, validation_data=test)

model.save('./unet.h5')
