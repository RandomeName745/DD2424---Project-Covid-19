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

###
config = parseConfig.parseConfig("config_augmentImages.json")
augmentor = createAugmentor(config["problem"],config["annotationMode"],config["outputMode"],config["generationMode"],config["inputPath"],
                            {"outputPath":config["outputPath"]})
transformer = transformerGenerator(config["problem"])
###

# Load the techniques and add them to the augmentor
techniques = [createTechnique(technique,param) for (technique,param) in config["augmentationTechniques"]]

i = 0

for technique in techniques:
    augmentor.addTransformer(transformer(technique))
    i = i + 1
    
agm = augmentor.applyAugmentation()
