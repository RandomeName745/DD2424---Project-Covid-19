#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:13:52 2020

@author: alex
"""
from __future__ import absolute_import
from matplotlib import pyplot as plt
# from keras.optimizers import SGD
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
from clodsa.utils.minivgg import MiniVGGNet
import cv2
import parseConfig

config = parseConfig.parseConfig("cats_dogs_folder_folder_linear.json")

augmentor = createAugmentor(config["problem"],config["annotationMode"],config["outputMode"],config["generationMode"],config["inputPath"],
                            {"outputPath":config["outputPath"]})

transformer = transformerGenerator(config["problem"])

# Load the techniques and add them to the augmentor
techniques = [createTechnique(technique,param) for (technique,param) in config["augmentationTechniques"]]

print("Number of images in input folder")
!ls /home/alex/anaconda3/envs/DD2424-project/DD2424---Project-Covid-19/git/datasets/images | wc -l

img = {}
img["original"] = cv2.imread(config["inputPath"] + "images/cat_1.jpg")


i = 0
plt.figure("original")
# changing to the BGR format of OpenCV to RGB format for matplotlib
plt.imshow(img["original"][:,:,::-1])
for technique in techniques:
    augmentor.addTransformer(transformer(technique))
    img[config["augmentationTechniques"][i][0]] = technique.apply(img["original"])
    plt.figure(config["augmentationTechniques"][i][0])
    plt.imshow(img[config["augmentationTechniques"][i][0]][:,:,::-1])
    i = i + 1
    
agm = augmentor.applyAugmentation()

print("Number of images in output folder")
!ls /home/alex/anaconda3/envs/DD2424-project/DD2424---Project-Covid-19/git/augmented_images/images | wc -l

