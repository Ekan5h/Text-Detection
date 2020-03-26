from keras.layers import Dense, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
import keras.backend as K
import config as cfg

maxHeight = cfg.maxHeight
maxWidth = cfg.maxWidth
charLen = cfg.charLen

def ctcLoss(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def getModel(maxLen):
    labels = Input(shape=[maxLen], dtype='float32', name='Labels')
    inpLen = Input(shape=[1], dtype='int64', name='InputLength')
    labelLen = Input(shape=[1], dtype='int64', name='LabelLength')
    inp = Input(shape=(maxWidth, maxHeight, 1))
    x = Conv2D(64, (3,3), activation = 'relu', padding='same', name='Conv1')(inp)
    x = MaxPool2D(pool_size=(2, 2), strides=2, name='Pool1')(x)
    x = Conv2D(128, (3,3), activation = 'relu', padding='same', name='Conv2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, name='Pool2')(x)
    x = Conv2D(256, (3,3), activation = 'relu', padding='same', name='Conv3')(x)
    x = Conv2D(256, (3,3), activation = 'relu', padding='same', name='Conv4')(x)
    x = MaxPool2D(pool_size=(2, 1), name='Pool4')(x)
    x = Conv2D(512, (3,3), activation = 'relu', padding='same', name='Conv5')(x)
    x = BatchNormalization(name='BatchNorm5')(x)
    x = Conv2D(512, (3,3), activation = 'relu', padding='same', name='Conv6')(x)
    x = BatchNormalization(name='BatchNorm6')(x)
    x = MaxPool2D(pool_size=(2, 1), name='Pool6')(x)
    x = Conv2D(512, (2,2), activation = 'relu', name='Conv7')(x)
    x = Lambda(lambda x: K.squeeze(x, 1), name='LambdaSqueeze')(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2, name='BiLSTM1'))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2, name='BiLSTM2'))(x)
    x = Dense(1+charLen, activation = 'softmax', name='Dense')(x)
    modelFront = Model(inp, x)
    outp = Lambda(ctcLoss, output_shape=(1,), name='CTCLoss')([x, labels, inpLen, labelLen])
    model = Model(inputs = [inp, labels, inpLen, labelLen], outputs= outp)
    model.compile(loss={'CTCLoss': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
    return [modelFront, model]