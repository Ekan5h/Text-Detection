from model import getModel
from dataset import getData
import config as cfg
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
tf.logging.set_verbosity(tf.logging.ERROR)

print('[LOADING TESTING DATA...]')
images, texts, _, _, _, _, maxLen, imageLocs = getData(cfg.testDataPath)

print('[LOADING MODEL...]')
model, _ = getModel(maxLen)
model.load_weights(cfg.finalModelPath)

print('[MAKING PREDICTIONS...]')
y_pred = model.predict(images)
p, q, _ = y_pred.shape
y_pred = K.ctc_decode(y_pred, np.ones(p)*q)
y_pred = y_pred[0][0]
y_pred = K.get_value(y_pred)
 
print('[DECODING PREDICTIONS...]')
predictions = []
for pred in y_pred:
    ans = ''
    for char in pred:  
        if int(char)<0:
            continue
        ans += cfg.chars[int(char)]
    predictions.append(ans)

print('[WRITING OUTPUT...]')
df = pd.DataFrame({"Image location":imageLocs, "Ground truth":texts, "Predictions":predictions})
df.index = np.arange(1, len(df)+1)
df.to_csv(cfg.outputPath, index=True)

     