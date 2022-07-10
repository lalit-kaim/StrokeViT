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
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from tensorflow.keras.applications.resnet import ResNet101
import json 
import glob
import shutil
     
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# creating model
model = ResNet50(weights='imagenet')                                                             # download the resnet50 model with imagenet weights
model.layers.pop()                                                                               # pop the last layer (model structure)
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)                              # pass inputs and outputs to new model(for weight files) 
model.trainable = False                                                                          # Freeze the outer model
model.summary()

# Extract features from image.
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))                                       # load image convert to size 224 x 224
    img_array = image.img_to_array(img)                                                          # convert image to array for preprocessing
    expanded_img_array = np.expand_dims(img_array, axis=0)                                       # expand dimentions of array since model takes input in 3D for convolution operation 
    preprocessed_img = preprocess_input(expanded_img_array)                                      # preprocess input which is imported from keras based on resnet50 model
    features = model.predict(preprocessed_img)                                                   # apply predictions
    flattened_features = features.flatten()                                                      # flatten features to 1D array 
    # print(flattened_features.shape)
    normalized_features = flattened_features / norm(flattened_features)                          # normalize features 
    return normalized_features
    # returns a vector of floating points i.e features we got after passing query imageto the model.


# def get_file_list():
#     totalFiles = []
#     files = glob.glob("./split_stroke/Normal-Train/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     files = glob.glob("./split_stroke/Normal-Test/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     files = glob.glob("./split_stroke/Normal-Validation/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     files = glob.glob("./split_stroke/Hemorrhage-Train/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     files = glob.glob("./split_stroke/Hemorrhage-Test/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     files = glob.glob("./split_stroke/Hemorrhage-Validation/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     files = glob.glob("./split_stroke/Infarct-Train/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     files = glob.glob("./split_stroke/Infarct-Test/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     files = glob.glob("./split_stroke/Infarct-Validation/*.jpg")
#     for f in files:
#         totalFiles.append(f)
#     print("TOTAL FILES : ",len(totalFiles))
#     return totalFiles        

# Binarizing the floating point numbers to 0 or 1 
# Simple step self explanatory class

class BinaryStep:
    def __init__(self, x):
        self.x = x

    def forward(self):
        self.x[self.x <= 0.00] = 0
        self.x[self.x > 0.00] = 1
        return self.x

    def backward(self):
        return self.x

# returns the binarized vector using the feature vector

''' Load pca object. '''
# pca object which is computed based on the images of caltech256 Object categories consisting of 30k images 
# code for computation is commented below:

# path to the datasets
#root_dir = '/path/256_ObjectCategories'

#Retrieving the filenames from this order
# def getPCA():
#     print("lalitkaim")
#     filenames = sorted(get_file_list())
#     feature_list = []
#     for i in tqdm_notebook(range(len(filenames))):
#         feature_list.append(extract_features(filenames[i]))
#     from sklearn.decomposition import PCA
#     num_feature_dimensions=128
#     pca = PCA(n_components = num_feature_dimensions)
#     pca.fit(feature_list)
#     feature_list_compressed = pca.transform(feature_list)
#     pickle.dump(pca,open('pca_128.pickle','wb'))
pca = pickle.load(open('./pca_128.pickle', 'rb'))
print("PCA : ",pca)


'''utility function to Convert hash-bits into a single hash string. conversion to string '''
# Self explanatory function
def convert(list): 
    s = [str(int(i)) for i in list] 
    mystring = ""
    for digit in s:
        mystring += str(int(digit))
    return(mystring)

# returns a binary string obtained from the binarized feture vector


def hamming2(x,y):
    """Utility function to Calculate the Hamming distance between two bit strings (fast method)"""
    assert len(x) == len(y)
    count,z = 0,int(x,2)^int(y,2)
    while z:
        count += 1
        z &= z-1 # this step reduces time complexity
    return count


# returns the hamming distance between strings passed to the function 


''' utility function generate the bit strings based on positions found by nCp formula '''
# we have bit string and the positions of the string to be flipped 
# we can produce strings after flipping the bits to generate the codes of given hamming distance

def get_codes_by_positions(string,positions):
    # print("Starting of get_codes_by_positions : ",string, positions)
    codes = []
    for i in range(len(positions)):
        a = string
        for j in range(len(positions[i])):
            b = a[0:positions[i][j]]
            c = a[positions[i][j] + 1:]
            b = b + str(1 - int(a[positions[i][j]]))
            a = b + c
        codes.append(a)
    return codes
# returns the codes we get after flipping the bits



# this function generates the positions to be flipped based on the hamming distance using permutation and combinations
# since we are using a 16 bit part we have to compute various positions for a code length of 16 and hamming distance = 3
positions = []
distance = 3
'''get positions based on nCp formula always fixed'''
def get_positions(dis):
    for i in range(dis+1):
        comb = combinations([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],i) 
        for j in list(comb):
            positions.append(j)
    return positions
positions = get_positions(distance)
# print(len(positions))
# returns the positions that needs to be flipped for a hamming distance


"""# Creating Mongo DB database"""

import pymongo
from pymongo.collation import Collation
import locale

client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
database = client['HashCodes']                 # name of database = HashCodes 
collection = database.object_categories     # name of collection = collection

''' queries the database and returns the hashcode with given part value and part number (i.e 0 yo 7)'''
def find_hashcodes(part_value,part):
    Ids = []
    Ids.extend(collection.distinct( "hashcode", { part : { "$in": part_value}} ))
    return Ids
# returns object ids obtained as a result of the query


''' inserting records in database based on given query=128 bit hashcode'''   
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
            'img_address':address   # address of the image
            }
    collection.insert_one(data)
    print('inserted record')

# collection.find({part:code},{'part0':0}).explain()['executionStats']

"""# Creating index on json files""" # needs to run only once from in0 to in7 

# in0 = database.object_categories.create_index([('parts0',1)])
# in1 = database.object_categories.create_index([('parts1',1)])
# in2 = database.object_categories.create_index([('parts2',1)])
# in3 = database.object_categories.create_index([('parts3',1)])
# in4 = database.object_categories.create_index([('parts4',1)])
# in5 = database.object_categories.create_index([('parts5',1)])
# in6 = database.object_categories.create_index([('parts6',1)])
# in7 = database.object_categories.create_index([('parts7',1)])
# in8 = database.object_categories.create_index([('hashcode',1)])

# database.collection.ensure_index([('part0',1)])

# database.object_categories.drop_indexes()
database.object_categories.index_information()
# database.collection.find({'parts0':'0000000000000000'},{})

"""# Processing Querry Image"""

def processing_query(query_path):
    features_singleNew = []
    features = extract_features(query_path)   # Extracting features of query image 
    x = features.reshape(1,-1)
    # feature_querry = pca.transform(x)   # PCA for single image 
    features_to_bin = BinaryStep(pca.transform(x))     # applying pca
    features_for = features_to_bin.forward()         # Converting features to binary 
    features_singleNew.append(convert(features_for[0])) # Appending features for predicting hamming distance (one image)
    return features_singleNew  # 128 bit vector (binary)

def stage_one_search(features_singleNew):
    # print("Start of stage one search : ", features_singleNew)
    st = time.time()
    z = 0
    hashcodes = []
    codes_to_query = [[0 for x in range(0)] for y in range(8)]
    # print(codes_to_query)
    # getting codes with given hamming distance for each part of the hashcode
    for j in range(0,len(features_singleNew[0]),16):
        k = features_singleNew[0][j:j+16]
        # print(k)
        codes_to_query[z].extend(get_codes_by_positions(k,positions))
        # print('length_of_codes_to_query',len(codes_to_query[z]))
        z = (z + 1) % 8
    ed = time.time()
    print('find codes time : ',ed-st)

    #Collecting hashcodes for the result images
    st1 = time.time()
    first = 'parts0'
    num = '0'
    for i in range(0,8):
        print(len(codes_to_query[i]))
        # query the database and find the hashcodes, for all codes in codes to query 
        # for taking care of each part we pass the variable 'first' with part name in it and change with each iteration
        hashcodes.extend(find_hashcodes(codes_to_query[i],first))  # query to the database
        first = first[:-1]
        num = chr(ord(num) + 1)
        first = first + num
        

    #  Removing Duplicates from hashcodes   
    x = []    
    for i in hashcodes:
        if i not in x:
            x.append(i)  
        
    print('number of hashcodes is',len(x))
    ed1 = time.time()
    print('query time : ',ed1-st1)
    # print("Value of x : ",x)
    return x           #all  hashcodes that are obtained as a result of search with no duplicates


def final_result(hashcodes, features_singleNew, query_path):
    # print(hashcodes, features_singleNew, query_path)
    cnt= [100,101,102,103,104] 
    address_list = []
    HC = ['0','1','0','1','0']
    for i in range(len(hashcodes)):
        count = hamming2(hashcodes[i],features_singleNew[0])  
        if(count<max(cnt)):
            maxpos = cnt.index(max(cnt)) 
            cnt.remove(max(cnt))
            cnt.append(count)
            HC.pop(maxpos)
            HC.append(hashcodes[i])
                           
    address_list.extend(collection.distinct( "img_address", { 'hashcode' : { "$in": HC}} ))

    #insertion
    #if same image is not found we insert the record in database 
    flag = 0
    for i in range(0,len(cnt)):
        if ( cnt[i] < 10 ):                           # if an image with hamming distance < 10 is found means image is same. 
            flag = 1
        
    # if(flag == 0):
    #     print('inserting')
    #     print(query_path)
    #     insert_record(features_singleNew[0],query_path)
        
    return address_list           # addresss after complete match 


# stores the top 5 images in the folder named results
def printing_results(address_list):
    # print('results:')
    N = 5
    final = address_list[-N:]

    results = os.listdir('static/results/')
    for f in results:
        os.remove(os.path.join('static/results/', f))
    # results = os.listdir('static/results/')
    # print('results',results)

    d = {"status":"success" ,"result_images": []}
    # my_dict = {"Name":[]}
    # d["status"] = ["success"]

    for i in range(5):
        # print(i)
        # img = cv2.imread(final[i])
        img = image.load_img(final[i])
        img = image.img_to_array(img)
        # file_ext = os.path.splitext(final[i])[1]
        file_name = os.path.split(final[i])
        # print(file_name[1])
        d['result_images'].append('static/results' + file_name[1])
        print("*"*10)
        image.save_img('static/results/' + file_name[1] , img)
        # cv2.imwrite('static/results/' + file_name[1] , img)
    return d 

#this method calls all the functions one by one

def call_search(file_path):
    features_singleNew =[]
    # src_dir = "/uploads"
    results = os.listdir('static/query/')
    for f in results:
        os.remove(os.path.join('static/query/', f))
    dst_dir = "static/query"
    for jpgfile in glob.iglob(os.path.join(file_path)):
        shutil.copy(jpgfile, dst_dir)
    features_singleNew = processing_query(file_path)
    # print("after : ", features_singleNew)
    hashcodes = stage_one_search(features_singleNew)
    print("hashcodes : ", len(hashcodes))
    address_list = final_result(hashcodes , features_singleNew , file_path)
    print("address list : ",address_list)
    d = printing_results(address_list)
    query = file_path[file_path.rindex("/")+1:]
    print("query"*5,query)
    print(file_path)
    return d 