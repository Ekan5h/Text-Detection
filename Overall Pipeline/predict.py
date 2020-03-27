from model import getModel
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
tf.logging.set_verbosity(tf.logging.ERROR)
def img2txt(image,model):
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    charLen = len(chars)
    y_pred = model.predict([image])
    p, q, _ = y_pred.shape
    y_pred = K.ctc_decode(y_pred, np.ones(p)*q)
    y_pred = y_pred[0][0]
    y_pred = K.get_value(y_pred)

    predictions = []
    for pred in y_pred:
        ans = ''
        for char in pred:  
            if int(char)<0:
                continue
            ans += chars[int(char)]
        predictions.append(ans)
    return predictions
