from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import glob 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import load_model
import os

def modelLoad(file):
    model = load_model("stroke_model_vgg16.h5")
    print("Model imported successfully")
    modelTest(file)
    
def modelTest(file):
    img = image.load_img(file, target_size=(224, 224))
    img = image.img_to_array(img)
    array=np.array(img)
    x_test=preprocess_input(array)
    model = load_model("stroke_model_vgg16.h5")
    model.summary()
    x_test = np.expand_dims(x_test, axis=0)
    y_pred = model.predict(x_test)
    ypred = np.argmax(y_pred, axis=1)
    if ypred==0:
        print("Hamorrhage")
    elif ypred==1:
        print("Infarct")
    elif ypred==2:
        print("Normal")
    else:
        print("Prediction unknown")
    fetchSimilarImage(ypred)
    

def fetchSimilarImage(ypred):
    combine_images = []
    images = []
    image_names = []
    # files = glob.glob("./combined/*.jpg")
    file_list = []
    root_dir = './split_stroke'
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
    files = sorted(file_list)
    for f1 in files[0:]: 
        img = image.load_img(f1, target_size=(224, 224))
        img_arr = image.img_to_array(img)
        images.append(img_arr)
        head, tail = os.path.split(f1)
        image_names.append(tail)
        combine_images.append(img_arr) 
    np_array=np.array(combine_images)
    pre_images=preprocess_input(np_array)
    model = load_model("stroke_model_vgg16.h5")
    pre_images, images, image_names = shuffle(pre_images, images, image_names)
    print(pre_images.shape)
    y_pred = model.predict(pre_images)
    combine_pred = np.argmax(y_pred, axis=1)
    output = []
    print(combine_pred)
    dir = 'static/results/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    count = 1
    for i, pred in enumerate(combine_pred):
        if ypred == pred:
            # print(i, images[i])
            image.save_img('static/results/'+image_names[i], images[i])
            count = count + 1
        if count > 5:
            break
