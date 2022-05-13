import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image
from model import model


# load model
try:
    #model = tf.keras.models.load_model(os.getcwd() + "/model/magenta_arbitrary-image-stylization-v1-256_2/")
    model = hub.load(os.getcwd() + "/model/magenta_arbitrary-image-stylization-v1-256_2/")
    # print("Model loaded")
except:
    print("Model not found")


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
      assert tensor.shape[0] == 1
      tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

# load data as list from temp folder

content_path = os.getcwd() + "/files/content/"
style_path = os.getcwd() + "/files/style/"

for filename in os.listdir(content_path):
    content_image = load_img(content_path + filename)
for filename in os.listdir(style_path):
    style_image = load_img(style_path + filename)

# perform model inference
file_name = 'result.jpeg'
result_path = os.getcwd() + "/files/result/"

stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image).save(result_path + file_name)

print("Style Transfer completed!")

