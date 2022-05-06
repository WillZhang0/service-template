import tensorflow as tf
import tensorflow_hub as hub
import os

# load model
try:
    #model = tf.keras.models.load_model(os.getcwd() + "/model/magenta_arbitrary-image-stylization-v1-256_2/")
    model = hub.load(os.getcwd() + "/model/magenta_arbitrary-image-stylization-v1-256_2/")
    print("Model loaded")
except:
    print("Model not found")
