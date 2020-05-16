#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 10:31:44 2020

@author: alex
"""
import argparse
from clodsa.utils.conf import Conf

def parseConfig(configfile):  
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
    args = vars(ap.parse_args(args=["-c",configfile]))   
    
    config = {}
    
    conf = Conf(args["conf"])

    config["problem"] = conf["problem"]
    config["annotationMode"] = conf["annotation_mode"]
    config["outputMode"] = conf["output_mode"]
    
    config["generationMode"] = conf["generation_mode"]
    config["inputPath"] = conf["input_path"]
    # parameters = conf["parameters"]
    config["outputPath"] = conf["output_path"]
    config["augmentationTechniques"] = conf["augmentation_techniques"]
    return config