import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from glob import glob

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

"""## Data Generator"""

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        self.image_path=sorted(glob(f"{self.path}/*.png"))
        # self.image_path = sorted(glob(f"{self.path}/image/*.png"))
        # self.mask_path = sorted(glob(f"{self.path}/mask/*.png"))
        # temp = list(zip(self.image_path, self.mask_path))
        # random.shuffle(temp)
        # self.image_path, self.mask_path = zip(*temp)
        
    def __load__(self, id_name):
        image_path = self.image_path
        # mask_path = self.mask_path
        
        image_ind = image_path[id_name]
        # mask_ind = mask_path[id_name]
        
        
        image = cv2.imread(image_ind, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # mask = cv2.imread(mask_ind, -1)
        # mask = cv2.resize(mask, (self.image_size, self.image_size))
        # mask = np.expand_dims(mask, axis=-1)
        
        image = image/255.0
        # mask = mask/255.0

        return image
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img = self.__load__(id_name)
            image.append(_img)
            # mask.append(_mask)
            
        image = np.array(image)
        # mask  = np.array(mask)
        
        return image
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))


image_size = 128
train_path = "./hem27cases/Train"
epochs = 5
batch_size = 8

# train_ind = []
# test_ind = []
# for i in range(1184):
#     train_ind.append(i);
    
# for i in range(448):
#     test_ind.append(i);
    
# train_ids = train_ind
# test_ids = test_ind

# print(train_path, image_size)
# gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
# x, y = gen.__getitem__(0)
# print(x.shape, y.shape)

# r = random.randint(0, len(x)-1)

# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(x[r])
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(np.reshape(y[r], (image_size, image_size)), cmap="gray")


# def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
#     c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
#     p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
#     return c, p

# def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
#     us = keras.layers.UpSampling2D((2, 2))(x)
#     concat = keras.layers.Concatenate()([us, skip])
#     c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
#     c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
#     return c

# def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
#     c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
#     return c


# def UNet():
#     f = [16, 32, 64, 128, 256]
#     inputs = keras.layers.Input((image_size, image_size, 3))
    
#     p0 = inputs
#     c1, p1 = down_block(p0, f[0]) #128 -> 64
#     c2, p2 = down_block(p1, f[1]) #64 -> 32
#     c3, p3 = down_block(p2, f[2]) #32 -> 16
#     c4, p4 = down_block(p3, f[3]) #16->8
    
#     bn = bottleneck(p4, f[4])
    
#     u1 = up_block(bn, c4, f[3]) #8 -> 16
#     u2 = up_block(u1, c3, f[2]) #16 -> 32
#     u3 = up_block(u2, c2, f[1]) #32 -> 64
#     u4 = up_block(u3, c1, f[0]) #64 -> 128
    
#     outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
#     model = keras.models.Model(inputs, outputs)
#     return model

# model = UNet()
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
# model.summary()

# """## Training the model"""

# train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
# # valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

# # print(len(train_gen[0][0][0]))

# train_steps = len(train_ids)//(batch_size)
# # valid_steps = len(valid_ids)//(batch_size)

# print(len(train_gen), train_steps)
# model.fit(train_gen, steps_per_epoch=train_steps, epochs=epochs)
# model.save('my_model.h5')

# test_path = "./hem27cases/Test"
# valid_ids = test_ids
# valid_gen = DataGen(valid_ids, test_path, image_size=image_size, batch_size=1)
# x_test = []
# y_test = []
# for x in valid_gen:
#     x_test.append(x[0][0])
#     y_test.append(x[1][0])

def segmentationFun(file_list):
    test_path = "./static/segmentation/input/"
    path, dirs, files = next(os.walk(test_path))
    file_count = len(files)
    print("file count : ",file_count)
    valid_ids = []
    for i in range(file_count):
        valid_ids.append(i)
    print(valid_ids)
    valid_gen = DataGen(valid_ids, test_path, image_size=image_size, batch_size=1)
    print(valid_gen)
    y_test = []
    for x in valid_gen:
        y_test.append(x[0])

    model = keras.models.load_model("./hem27stroke.h5")

    y_pred = model.predict(valid_gen)

    print(y_pred)
    for i in y_test:
        print(i)
    y_pred = y_pred > 0.5
    y_pred = y_pred + 0
    y_pred = y_pred*255.0
    for i, img in enumerate(y_pred):
        cv2.imwrite("./static/segmentation/output/"+file_list[i], img)