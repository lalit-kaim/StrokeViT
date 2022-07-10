import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.linalg import norm
import pickle
from tqdm import tqdm, tqdm_notebook
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import time
from itertools import combinations 
import threading


from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from tensorflow.keras.applications.resnet import ResNet101

model = ResNet50(weights='imagenet')
# model1 = ResNet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

model.summary()


from tensorflow.keras.preprocessing import image

# Extract features from image.
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
#     img = image.load_img(img_path, target_size=(96, 112))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    # print(flattened_features.shape)
    print("lalit   ")
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features 


# Get all the extensions of all the filenames in the dataset.

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list



# path to the datasets
root_dir = './split_stroke'

#Retrieving the filenames from this order
filenames = sorted(get_file_list(root_dir))
for i in range(len(filenames)):
    print(filenames[i],i)


''' Get all the features in the database for all filenames.'''
feature_list = []
var = 1
for i in tqdm_notebook(range(len(filenames))):
    feature_list.append(extract_features(filenames[i], model))


class BinaryStep:
    def __init__(self, x):
        self.x = x

    def forward(self):
        self.x[self.x <= 0.00] = 0
        self.x[self.x > 0.00] = 1
        return self.x

    def backward(self):
        return self.x

    
''' Load pca object. '''
pca = pickle.load(open('./pca_128.pickle', 'rb'))
print(pca)


'''applying PCA  and converting to binary '''
features_to_binary = BinaryStep(pca.transform(feature_list))
features_forward = features_to_binary.forward()
print(features_forward.shape)

''' Execute this only when new PCA needs to be computed (when a large number of images are added to database)'''

# ''' PCA is applied to get the number of dimensions to get hashcodes.'''
from sklearn.decomposition import PCA
num_feature_dimensions=128
pca = PCA(n_components = num_feature_dimensions)
pca.fit(feature_list)

features_to_binary = BinaryStep(pca.transform(feature_list))
features_forward = features_to_binary.forward()

''' Hash codes. '''
print(features_forward.shape)

''' you can dump the file and use it later'''


'''Convert hash-bits into a single hash code. conversion to string '''
def convert(list): 
    s = [str(int(i)) for i in list] 
    mystring = ""
    for digit in s:
        mystring += str(int(digit))
    return(mystring) 


import pymongo
from pymongo.collation import Collation
import locale

client = pymongo.MongoClient('mongodb://127.0.0.1:27017')


database=client['HashCodes']            #name of database = hashcodes  

collection=database.object_categories             #name of collection = codes




def insert_record(query,address):

    data = {'hashcode':query,
            'parts0':query[0:16],
            'parts1':query[16:32],
            'parts2':query[32:48],
            'parts3':query[48:64],
            'parts4':query[64:80],
            'parts5':query[80:96],
            'parts6':query[96:112],
            'parts7':query[112:128],
            'img_address':address
            }
    collection.insert_one(data)



in0 = database.object_categories.create_index([('parts0',1)])
in1 = database.object_categories.create_index([('parts1',1)])
in2 = database.object_categories.create_index([('parts2',1)])
in3 = database.object_categories.create_index([('parts3',1)])
in4 = database.object_categories.create_index([('parts4',1)])
in5 = database.object_categories.create_index([('parts5',1)])
in6 = database.object_categories.create_index([('parts6',1)])
in7 = database.object_categories.create_index([('parts7',1)])



# database.collection.drop_indexes()


database.object_categories.index_information()

# database.collection.find({'parts0':'0000000000000000'},{})


'''New features list consisting of the appended hashcodes'''
features_new = []
for i in tqdm_notebook(range(len(feature_list))):
    features_new.append(convert(features_forward[i]))


''' calling insert funtion '''
for i in range (0,len(filenames)):
    insert_record(features_new[i],filenames[i])
    print("inserting",i)
