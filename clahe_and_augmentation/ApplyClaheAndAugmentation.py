#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:37:00 2020

@author: alex
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import AugmentationTechniques as AT
import cv2
import os
import glob
import tensorflow_datasets as tfds


doCLAHE = 0
doAugmentation = 1

def CallAugmentationFunctions(tech):
    if tech == 'zoom':
        augmentation = AT.zoom
    elif tech == 'rotate':
        augmentation = AT.rotate
    elif tech == 'contrast':
        augmentation = AT.contrast
    elif tech == 'brightness':
        augmentation = AT.brightness
    elif tech == 'flip':
        augmentation = AT.flip
    return augmentation

def applyCLAHE(dirName, dirDataset, folderCatergory, imgList, clipLim, tilegridSize):
    """
    Histogram Equalization considers the global contrast of the image, may not give good results.
    Adaptive histogram equalization divides images into small tiles and performs hist. eq.
    Contrast limiting is also applied to minimize aplification of noise.
    Together the algorithm is called: Contrast Limited Adaptive Histogram Equalization (CLAHE)
    """
    
    folderClahe = 'dataset_clahe/'
    # Start by creating a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=clipLim, tileGridSize=(tilegridSize,tilegridSize))  #Define tile size and clip limit. 
    
    for i in range(len(imgList)):
    # for i in range(3):
        imgName = os.path.basename(dirName + imgList[i])
        # Read image
        img = cv2.imread(dirName + imgName)       
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        # Apply CLAHE and save in clahe-folder
        cl = np.zeros((224,224,3))
        for j in range(img.shape[2]):            
            cl[:,:,j] = clahe.apply(img[:,:,j])
        
        if os.path.isdir(dirDataset + folderClahe) == False:
            os.mkdir(dirDataset + folderClahe)
        if os.path.isdir(dirDataset + folderClahe + folderCatergory) == False:
            os.mkdir(dirDataset + folderClahe + folderCatergory)
        cv2.imwrite(dirDataset + folderClahe + folderCatergory + '/' + imgName, cl) 
        
def applyAugmentation(dirName, dirDataset, folderCatergory, imgList, augmentation_techniques):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    folderAugmentation = 'dataset_clahe_augmentedTF/'
    
    for i in range(len(imgList)):
    # for i in range(10):
        imgName = os.path.basename(dirName + imgList[i])
        img = cv2.imread(dirName + imgName).astype(np.float32)#/255
        img = np.reshape(img,(1, img.shape[0], img.shape[1], img.shape[2]))
        img = tf.data.Dataset.from_tensor_slices(img)
        if os.path.isdir(dirDataset + folderAugmentation) == False:
            os.mkdir(dirDataset + folderAugmentation)
        if os.path.isdir(dirDataset + folderAugmentation + folderCatergory) == False:
            os.mkdir(dirDataset + folderAugmentation + folderCatergory)
        for key in augmentation_techniques:
            # Call augmentation function
            f = CallAugmentationFunctions(key)
            img_agm = 0
            # img_agm = img.map(lambda x: tf.cond(lambda: f(x), lambda: x), num_parallel_calls=4)
            img_agm = img.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0, lambda: f(x), lambda: x), num_parallel_calls=4)
            # img_agm = img_agm.map(lambda x: tf.clip_by_value(x, 0, 1))                
            img_agm=tfds.as_numpy(img_agm, graph=None)
            img_agm=np.array(list(img_agm))
            img_agm = np.reshape(img_agm,(img_agm.shape[1], img_agm.shape[2], img_agm.shape[3]))#*255
            cv2.imwrite(dirDataset + folderAugmentation + folderCatergory + '/' + imgName + key + '.jpg' , img_agm) 
    

# Insert the path to the locally stored datasets
# last level of path (here datasets) initally contains a single subfolder in 
# which the raw data is stored: dirRawData, other subfolders for the processed 
# data are created
dirDataset = '/media/alex/shared/documents/Uni/Masterstudium/Auslandsstudium/DD2424/project/datasets/'
dirRawData = 'dataset_raw/'
dirClaheData = 'dataset_clahe/'

###############################################################################
if doCLAHE:
    # Get the classification subfolders
    folderList = os.listdir(dirDataset + dirRawData)
    
    for dd in range(len(folderList)):
        dirName = dirDataset + dirRawData + folderList[dd] + '/'
        # For the given path, get the List of all images in the directory tree 
        imgList = glob.glob(dirName + '*.jpeg')
        imgList.extend(glob.glob(dirName + '*.png'))
        imgList.extend(glob.glob(dirName + '*.jpg'))
        # Apply CAHE
        applyCLAHE(dirName, dirDataset, folderList[dd], imgList, clipLim = 4, tilegridSize = 9)
    
 #############################################################################   
if doAugmentation:    
    augmentation_techniques = {'zoom', 'flip', 'contrast', 'brightness'}
    
    images = {'original': []}
    
    # Get the classification subfolders
    folderList = os.listdir(dirDataset + dirClaheData)
    #
    for dd in range(len(folderList)):
        dirName = dirDataset + dirClaheData + folderList[dd] + '/'
        # For the given path, get the List of all images in the directory tree 
        imgList = glob.glob(dirName + '*.jpeg')
        imgList.extend(glob.glob(dirName + '*.png'))
        imgList.extend(glob.glob(dirName + '*.jpg'))
        applyAugmentation(dirName, dirDataset, folderList[dd], imgList, augmentation_techniques)
    

