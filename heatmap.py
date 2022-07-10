import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.layers import Flatten, Dense, BatchNormalization, Activation,Dropout
from keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from numpy import load

# mymodel = Sequential()
# mymodel.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# mymodel.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# mymodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# mymodel.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# mymodel.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# mymodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# mymodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# mymodel.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# mymodel.add(Dropout(0.3))
# mymodel.add(Flatten()) 
# mymodel.add(Dropout(0.3))
# mymodel.add(Dense(3, activation=('softmax')))
# opt = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9)
# mymodel.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, heatmap_img_name="heatmap.jpg", alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save("./static/heatmap/"+heatmap_img_name)

def getheatmap(file):
    print("heatmap : ",file)
    mymodel = keras.models.load_model('heatmap_model.h5')
    # X = load('x_test.npy')
    # Y = load('y_test.npy')
    # acc = mymodel.evaluate(X, Y)
    # print(acc)
    mymodel.summary()
    img = image.load_img(file, target_size=(224, 224))
    img = image.img_to_array(img)
    array=np.array(img)
    x_test=preprocess_input(array)
    x_test = np.expand_dims(x_test, axis=0)
    y_pred = mymodel.predict(x_test)
    ypred = np.argmax(y_pred, axis=1)
    imgg = get_img_array(file, (224,224))
    last_conv_layer_name = "max_pooling2d_4"
    print(imgg.shape)
    heatmap = make_gradcam_heatmap(imgg, mymodel, last_conv_layer_name)
    # plt.matshow(heatmap)
    # plt.show()

    img = image.load_img(file)
    img = image.img_to_array(img)
    image.save_img('static/heatmap/1_original.jpg', img)

    save_and_display_gradcam(file, heatmap)
    if ypred==0:
        return "Haemorrhage"
    elif ypred==1:
        return "Infarct"
    elif ypred==2:
        return "Normal"