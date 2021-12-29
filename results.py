import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize

import PIL.Image

from model import IoU

IMAGE_FILEPATHS = [
	'test_images/1.jpeg', 
	'test_images/2.jpg',
	'test_images/3.jpg',
	'test_images/4.webp',
	'test_images/5.webp',
	'test_images/6.jpg',
]

def multiple_of_32(p):
	"""
	Turn image into dimensions that are multiples of 32 to be fed into the U-net
	"""
	x, y = p
	factor = y / x

	if factor < 1:
		x = 128
		y = int((factor * 128) // 32 * 32)
	else:
		y = 128
		x = int((1 / factor * 128) // 32 * 32)

	return (x, y)

kwargs = {
	'custom_objects': {'IoU': IoU}
}

model1 = tf.keras.models.load_model('unet.h5', **kwargs)

def predict(model):
	original_images = []
	images = []
	for i in IMAGE_FILEPATHS:
		original_image = PIL.Image.open(i)
		original_array = np.array(original_image)
		resized_image = original_image.resize((128, 128))

		resized_array = np.array(resized_image) / 255.
		resized_array = resized_array.reshape((1, resized_array.shape[0], resized_array.shape[1], 3))

		pred = model.predict(resized_array)

		pred = (pred >= 0.5).astype(np.uint8)[0]
		resized_pred1 = resize(pred, (original_array.shape[0], original_array.shape[1], 1))

		new_image_array = (resized_pred1 * original_array * 255).astype(np.uint8)

		new_image = PIL.Image.fromarray(new_image_array)
		
		original_images.append(original_image)
		images.append(new_image)

	return original_images, images

original_images, images = predict(model1)
plt.figure(figsize=(6, 6))
plt.subplot(3, 2, 1)
plt.imshow(original_images[0])
plt.axis('off')
plt.subplot(3, 2, 2)
plt.imshow(images[0])
plt.axis('off')
plt.subplot(3, 2, 3)
plt.imshow(original_images[1])
plt.axis('off')
plt.subplot(3, 2, 4)
plt.imshow(images[1])
plt.axis('off')
plt.subplot(3, 2, 5)
plt.imshow(original_images[3])
plt.axis('off')
plt.subplot(3, 2, 6)
plt.imshow(images[3])
plt.axis('off')
# for i in range(len(images)):
# 	plt.subplot(len(images) // 2, 2, i + 1)
# 	plt.imshow(images[i])
# 	plt.axis('off')
plt.show()
