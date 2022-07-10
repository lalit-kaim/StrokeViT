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
    file.sort()
    model = load_model("stroke_250cases_RESNET50.h5")
    print("Model imported successfully")
    return modelTest(file)
    
def modelTest(file):
    file.sort()
    test_images = []
    print(file)
    for i in range(len(file)):
        img = image.load_img(file[i], target_size=(224, 224))
        img = image.img_to_array(img) 
        test_images.append(img) 
    array=np.array(test_images)
    x_test=preprocess_input(array)
    model = load_model("stroke_250cases_RESNET50.h5")
    model.summary()
    print(x_test.shape)
    y_pred = model.predict(x_test)
    ypred = np.argmax(y_pred, axis=1)
    print(ypred)
    print(tuple(zip(file, ypred)))
    prediction = []
    for x in ypred:
        if x==0:
            prediction.append("Haemorrhage")
        elif x==1:
            prediction.append("Infarct")
        else:
            prediction.append("Normal")
    return prediction

def getCaseClassify(files):
    # files.sort()
    print(files)
    return modelLoad(files)