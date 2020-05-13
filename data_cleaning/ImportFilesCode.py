#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:48:07 2020

@author: alex
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import PIL 
from PIL import Image
import IPython.display as display
import os
from IPython.display import clear_output
 
def GetData(link,fname):
    # X-ray images
    data_dir = tf.keras.utils.get_file(origin=link,
                                      fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)
    # Return path for directory
    return data_dir
 
def CheckPathContent(input_dir):
    image_count = len(list(input_dir.glob('*/*.jpeg')))
    print(image_count)
 
    class_names = np.array([item.name for item in input_dir.glob('*')])
    print(class_names)
    return image_count, class_names
 
def ViewImage(classification,img_nr):
    pneumonia = list(data_dir.glob('PNEUMONIA/*'))
    image_path = pneumonia[1]
    display.display(Image.open(str(image_path)))
 
def get_label(file_path):
    # Path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == class_names
 
def decode_img(img):
    # Compressed string to 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # To floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [224,224])
    
def Preprocess(file_path):
    # With label func()
    label = get_label(file_path)
 
    # With img func()
    # load raw data from file as string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label
 
 
AUTOTUNE = tf.data.experimental.AUTOTUNE
 
# Saves to keras/datasets/fname
data_dir = GetData('https://drive.google.com/u/0/uc?id=1_2wCFEfYbC33C-eWAbqkM65zgrGZb_fh&export=download'
                    ,'Test_Data')
# test_data_dir_test = GetData('https://drive.google.com/u/0/uc?id=1_2wCFEfYbC33C-eWAbqkM65zgrGZb_fh&export=download'
#                    ,'Test_Data')
 
# Checks the Imported Data
image_count, class_names = CheckPathContent(data_dir)
#test_image_count, test_class_names = CheckPathContent(test_data_dir)
 
# Creating Datasets of the File Paths w/ Dataset.list_files() 
list_datasets = tf.data.Dataset.list_files(str(data_dir/'*/*'))
# print(' ')
# print('Dataset path list:')
# for f in list_datasets.take(5):
#     print(f.numpy())
 
# Preprocess each img in list_datasets
labeled_datasets = list_datasets.map(Preprocess, num_parallel_calls=AUTOTUNE)
 
# print(' ')
# print('What the labeled_datasets is:')
#labeled_datasets 

## Unpack Images and Labels
#  images = []
#  labels = []
#  for img,label in labeled_datasets:
#    images.append(img)
#    labels.append(label)

#image.numpy().shape

# # Check Images and Labels
#  plt.imshow(images[11].numpy())
#  print(labels[0].numpy())

