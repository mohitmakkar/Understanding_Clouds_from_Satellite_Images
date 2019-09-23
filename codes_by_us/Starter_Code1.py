import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
#%%
#path_code = os.path.dirname(os.path.realpath(__file__)) + '\\'

path_code ='E:\\Study\\Kaggle\\Understanding_Clouds_from_Satellite_Images\\codes_by_us\\'
path_data=path_code.replace('codes_by_us','sample_train')

def load_img(file_path):
    img = cv2.imread(file_path)
#    .astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    
def convert_to_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def display_img_as_gray(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')    
#%%
#img = load_img(path_data+'6a6b274.jpg')
#img = convert_to_gray(img)
#display_img(img)
#display_img_as_gray(img)

#%%
#Reading Csv file
train_df = pd.read_csv(path_data+"train.csv")

#Reading all images
train_image_paths = sorted(glob.glob(path_data + '*.jpg'))
train_images = np.array([load_img(file) for file in train_image_paths])
print(train_images.shape)

#%%
#Creating Dictionaries for image_pixel & Image_Class

train_df.dropna(inplace=True) #Dropping Labels not found in image
class_label_list = ["Gravel","Sugar","Fish","Flower"]

img_pixel_dict = {}
img_label_dict = {}
for idx, row in train_df.iterrows():
    
    img_name = row.Image_Label.split("_")[0]
    img_label = row.Image_Label.split("_")[1]
    
    img_label_encoded = class_label_list.index(img_label)

    if img_pixel_dict.get(img_name):
        img_pixel_dict[img_name].append(row.EncodedPixels)
        img_label_dict[img_name].append(img_label_encoded)
    else:
        img_pixel_dict[img_name] = [row.EncodedPixels]
        img_label_dict[img_name] = [img_label_encoded]

#%%













#
#
#
##%%
#
#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
#
##%%
#model = Sequential()
#
### FIRST SET OF LAYERS
#
## CONVOLUTIONAL LAYER
#model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
## POOLING LAYER
#model.add(MaxPool2D(pool_size=(2, 2)))
#
### SECOND SET OF LAYERS
#
## CONVOLUTIONAL LAYER
#model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
## POOLING LAYER
#model.add(MaxPool2D(pool_size=(2, 2)))
#
## FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
#model.add(Flatten())
#
## 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
#model.add(Dense(256, activation='relu'))
#
## LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
#model.add(Dense(10, activation='softmax'))
#
#
#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#
##%%
#
#model.fit(x_train,y_cat_train,verbose=1,epochs=10)
#
#
#model.evaluate(x_test,y_cat_test)
#
#
#
#from sklearn.metrics import classification_report
#
#predictions = model.predict_classes(x_test)
#
#
#
#print(classification_report(y_test,predictions))