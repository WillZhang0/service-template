import os
import numpy as np
import io
import random
import glob
import cv2
import tensorflow as tf
import PIL.Image
from model import model


def run(jobID):
  """
  title:: 
      run
  description:: 
      Run the model/get the predictions according the service.
  inputs::
      jobID 
            Job ID from datashop application used for search file or save file

  returns::
      insightsDataFileLocation
      insights data file location.

  load data from temp folder
    >  json data is data.json
    >  all images and CSV are named with jobID_"filetype"
    >  jobiD_csv.csv   "61ef72ed396fc5330c15f250_csv.csv"
    >  jobiD_image.png   "61ef72ed396fc5330c15f250_image.png"
  """
  # load data as list from temp folder

  fileslist = glob.glob(os.getcwd() + "/tmp/" + jobID + "-image"+"*")

  # perform model inference

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

  content_path = fileslist[0]
  style_path = fileslist[1]
  content_image = load_img(content_path)
  style_image = load_img(style_path)

  stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

  # for multiple results return list of resutls   results = [result_1,result_2,result_3]

  print("Style Transfer completed!")

  return [tensor_to_image(stylized_image)]
