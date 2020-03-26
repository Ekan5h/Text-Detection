import os
import numpy as np
import cv2
import fnmatch
import config as cfg
from keras.preprocessing.sequence import pad_sequences
maxHeight = cfg.maxHeight
maxWidth = cfg.maxWidth
inputLen = cfg.inputLen
chars = cfg.chars

def encode(text):
    encoded = []
    for char in text:
        encoded.append(chars.index(char))        
    return encoded

def preprocess(location):
    image = cv2.imread(location)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[0] > maxWidth or image.shape[1] > maxHeight:
        return []
    if image.shape[0] < maxWidth:
        image = np.concatenate((image, (np.ones((maxWidth - image.shape[0], image.shape[1]))*255)))
    if image.shape[1] < maxHeight:
        image = np.concatenate((image, (np.ones((maxWidth, maxHeight - image.shape[1]))*255)), axis=1)
    image = np.expand_dims(image , axis=2)
    image = image/255.0
    return image

def getData(path):
    images = []
    imageLocs = []
    texts = []
    encodedTexts = []
    inputLens = []
    labelLen = []
    maxLen = 0
    for base, _, files in os.walk(path):
        for imageName in fnmatch.filter(files, '*.jpg'):
            location = (os.path.join(base, imageName)).replace('\\', '/')
            image = preprocess(location)
            if image == []:
                continue
            text = imageName.split('_')[1]
            maxLen = max(maxLen, len(text))
            images.append(image)
            imageLocs.append(location)
            texts.append(text)
            encodedTexts.append(encode(text)) 
            labelLen.append(len(text))
            inputLens.append(inputLen)
    images = np.array(images)
    inputLens = np.array(inputLens)
    labelLen = np.array(labelLen)
    paddedTexts = pad_sequences(encodedTexts, maxlen=maxLen, padding='post', value = len(chars))
    return [images, texts, encodedTexts, labelLen, inputLens, paddedTexts, maxLen, imageLocs]