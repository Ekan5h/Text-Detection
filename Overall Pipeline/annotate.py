import os
import numpy as np
import cv2
import sys
import imutils
from model import PixelLink4s, getModel
from predict import img2txt
from utils import *

r_mean = 123.
g_mean = 117.
b_mean = 104.
rgb_mean = [r_mean, g_mean, b_mean]

model1, _ = getModel(64)
model1.load_weights('CRNN.h5')

img_path = sys.argv[1]
save_weights = r'pixellink.h5'
model = PixelLink4s((224,224,3))
model.load_weights(save_weights)
image = cv2.imread(img_path)
image = cv2.resize(image, (224, 224))
image_ori = image.copy()
image = image[...,::-1] - rgb_mean
image = np.expand_dims(image, axis=0)

pixel_pos_scores, link_pos_scores = model.predict(image)
pixel_pos_scores = softmax(pixel_pos_scores, axis=-1)
link_pos_scores_reshaped = link_pos_scores.reshape(link_pos_scores.shape[:-1]+(8, 2))
link_pos_scores = softmax(link_pos_scores_reshaped, axis=-1)

masks = decode_batch(pixel_pos_scores, link_pos_scores, pixel_conf_threshold=0.75, link_conf_threshold=0.9)

bboxes = mask_to_bboxes(masks[0], image_ori.shape)

image_c = image_ori.copy()
for box in bboxes:
    points = np.reshape(box, [4, 2])
    cv2.line(image_c,tuple(points[0]),tuple(points[1]),(0,0,255),2)
    cv2.line(image_c,tuple(points[0]),tuple(points[3]),(0,0,255),2)
    cv2.line(image_c,tuple(points[1]),tuple(points[2]),(0,0,255),2)
    cv2.line(image_c,tuple(points[2]),tuple(points[3]),(0,0,255),2)
    xstart = min(points[0][0],points[1][0],points[2][0],points[3][0])
    xend = max(points[0][0],points[1][0],points[2][0],points[3][0])
    ystart = min(points[0][1],points[1][1],points[2][1],points[3][1])
    yend = max(points[0][1],points[1][1],points[2][1],points[3][1])
    text = image_c[ystart:yend,xstart:xend,:]
    if(int((xend-xstart)*32/(yend-ystart))<=128):
        text = cv2.resize(text, (int((xend-xstart)*32/(yend-ystart)),32))
        text = cv2.copyMakeBorder(text,0,0,0,128-int((xend-xstart)*32/(yend-ystart)), cv2.BORDER_CONSTANT)
    else:
        text = cv2.resize(text, (128,int((yend-ystart)*128/(xend-xstart))))
        text = cv2.copyMakeBorder(text,0,32-int((yend-ystart)*128/(xend-xstart)),0,0, cv2.BORDER_CONSTANT)
    text = img2txt([cv2.cvtColor(text, cv2.COLOR_BGR2GRAY).reshape(32,128,1)],model1)
    if(len(text)>0):
        cv2.putText(image_c,text[0],(points[0][0],points[0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2,cv2.LINE_AA) 
cv2.imshow('image', image_c)
cv2.waitKey(0)
cv2.destroyAllWindows()

