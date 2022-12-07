#imports
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
from tqdm import tqdm
from keras.applications import xception

from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Input
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd
import numpy as np
import datetime
import scipy

SAVE_PATH = 'images/'
selected_breed_list = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog', 'shih-tzu', 'great_pyrenees', 'pomeranian', 'basenji', 'samoyed', 'airedale', 'tibetan_terrier']
model = load_model( 'src/model/2022-12-04_dog_breed_model.h5')


#using the model to detect the dog breed on image
def predict_from_image(img_path):
  img = load_img(img_path, target_size=(299, 299))
  img_tensor = img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)
  img_tensor /= 255.

  pred = model.predict(img_tensor)
  sorted_breeds_list = sorted(selected_breed_list)
  predicted_class = sorted_breeds_list[np.argmax(pred)]

  print(pred) 

  return predicted_class

app = Flask(__name__)
api = Api(app)

@app.route('/', methods = ['GET'])
def index():
  return {'success': True}

@app.route('/upload', methods = ['POST'])
def upload_file():
  if request.method == 'POST':
    #getting and saving the file
    f = request.files['file']
    filepath = SAVE_PATH + f.filename
    f.save(filepath)

    return predict_from_image(filepath)

if __name__ == '__main__':
  app.run(debug = True)
  print('server running')