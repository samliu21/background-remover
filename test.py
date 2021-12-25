import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize

import PIL.Image

IMAGE_FILEPATH = 'test_images/dog2.jpg'

def load_model():
	reloaded = tf.keras.models.load_model('./unet.h5')

	return reloaded

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


model = load_model()

original_image = PIL.Image.open(IMAGE_FILEPATH)
original_array = np.array(original_image)
resized_image = original_image.resize(multiple_of_32(original_image.size))

resized_array = np.array(resized_image)
resized_array = resized_array.reshape((1, resized_array.shape[0], resized_array.shape[1], 3))

pred = model.predict(resized_array)
pred = (pred >= 0.5).astype(np.uint8)[0]
resized_pred = resize(pred, (original_array.shape[0], original_array.shape[1], 1))

new_image_array = (resized_pred * original_array * 255).astype(np.uint8)

new_image = PIL.Image.fromarray(new_image_array)

plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(new_image)
plt.axis('off')
plt.show()

# new_image.show()
# new_image.save('img.png')
