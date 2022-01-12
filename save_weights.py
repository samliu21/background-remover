import tensorflow as tf

from model import IoU

model = tf.keras.models.load_model('znet37.h5', custom_objects={'IoU': IoU})
model.save_weights('weights')
