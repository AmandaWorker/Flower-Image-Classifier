# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers

#Make all other necessary imports.
import numpy as np
import json
from PIL import Image


import argparse

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('test_image_path')
parser.add_argument('reloaded_keras_model')
parser.add_argument('--top_k',type=int,default=5)
parser.add_argument('--category_names',default='label_map.json')

args = parser.parse_args()
print(args)
print('arg1:', args.test_image_path)
print('arg2:', args.reloaded_keras_model)
print('top_k:', args.top_k)
print('category_names:', args.category_names)
with open(args.category_names, 'r') as f:
     class_names = json.load(f)

test_image_path = args.test_image_path  
reloaded_keras_model = tf.keras.models.load_model(args.reloaded_keras_model ,custom_objects={'KerasLayer':hub.KerasLayer}, compile=False )
top_k = args.top_k

IMG_SHAPE  = 224

def process_image(image_array):
    image_tensor = tf.convert_to_tensor(image_array)
    image_resized = tf.image.resize(image_tensor, [IMG_SHAPE, IMG_SHAPE])
    image_resized /= 255
    image_final = image_resized.numpy()
    return image_final


def predict(test_image_path, model, top_k):
    im = Image.open(test_image_path)
    test_image = np.asarray(im)
    test_image_proc = process_image(test_image)
    expand_image = np.expand_dims(test_image_proc, axis=0)
    img_prediction = model.predict(expand_image)

    probs, classes = tf.math.top_k(img_prediction,top_k)
    probs = probs.numpy().squeeze()
    classes = classes.numpy().squeeze()
    #classes = [str(value) for value in classes]
    classes=[class_names[str(value+1)] for value in classes]
    return probs, classes   


probs, classes = predict(test_image_path, reloaded_keras_model, top_k)
np.set_printoptions(suppress=True)
print('Predicted Flower Name: \n',classes)
print('Probabilities: \n ', probs)