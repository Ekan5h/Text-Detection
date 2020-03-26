from model import getModel
from dataset import getData
import numpy as np
import config as cfg
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

print('[LOADING TRAINING DATA...]')
images, texts, encodedTexts, labelLen, inputLens, paddedTexts, maxLen, _ = getData(cfg.trainDataPath)

print('[CREATING MODEL...]')
_, model = getModel(maxLen)

print('[TRAINING MODEL...]')
model.fit(x = [images, paddedTexts, inputLens, labelLen],
            y = np.zeros(len(images)),
            batch_size = cfg.batch_size,
            epochs = cfg.epochs,
            verbose = 1)

print('[SAVING MODEL...]')
model.save_weights(cfg.trainingModelPath)

print('[TRAINING COMPLETE...]')