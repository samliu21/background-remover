import tensorflow as tf

from model import unet, IoU

img_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale=1./255,
)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

LOAD_EXISTING_MODEL = True

TRAIN_IMG_PATH = 'train_img'
TRAIN_MASK_PATH = 'train_mask'
VAL_IMG_PATH = 'val_img'
VAL_MASK_PATH = 'val_mask'

seed = 4
params = dict(
	target_size=(128, 128),
	class_mode=None,
	seed=seed,
)

train_img_gen = img_datagen.flow_from_directory(
	directory=TRAIN_IMG_PATH,
	**params,
)
train_mask_gen = mask_datagen.flow_from_directory(
	directory=TRAIN_MASK_PATH,
	color_mode='grayscale',
	**params,
)

val_img_gen = img_datagen.flow_from_directory(
	directory=VAL_IMG_PATH,
	**params,
)
val_mask_gen = mask_datagen.flow_from_directory(
	directory=VAL_MASK_PATH,
	color_mode='grayscale',
	**params,
)

train = zip(train_img_gen, train_mask_gen)
val = zip(val_img_gen, val_mask_gen)

if LOAD_EXISTING_MODEL:
	unet.load_weights('weights')

class SaveModel(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save('xnet{}.h5'.format(epoch + 1), overwrite=True,) 

cbk = SaveModel()

EPOCHS = 50

unet.fit(
	train, 
	steps_per_epoch=500, 
	epochs=EPOCHS, 
	validation_data=val,
	validation_steps=10,
	callbacks=[cbk],
)
