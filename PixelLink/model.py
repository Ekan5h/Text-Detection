import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Add, Lambda
import tensorflow as tf

def VGG16(input_shape=None, act = 'relu'):
    inp = Input(shape = input_shape)

    x = Conv2D(64, (3, 3), activation=act, padding='same', name="conv1_1")(inp)
    x = Conv2D(64, (3, 3), activation=act, padding='same', name="conv1_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)    
    
    x = Conv2D(128, (3, 3), activation=act, padding='same', name="conv2_1")(x)
    x = Conv2D(128, (3, 3), activation=act, padding='same', name="conv2_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)    
    
    x = Conv2D(256, (3, 3), activation=act, padding='same', name="conv3_1")(x)
    x = Conv2D(256, (3, 3), activation=act, padding='same', name="conv3_2")(x)
    x = Conv2D(256, (3, 3), activation=act, padding='same', name="conv3_3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)    
    
    x = Conv2D(512, (3, 3), activation=act, padding='same', name="conv4_1")(x)
    x = Conv2D(512, (3, 3), activation=act, padding='same', name="conv4_2")(x)
    x = Conv2D(512, (3, 3), activation=act, padding='same', name="conv4_3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)    
    
    x = Conv2D(512, (3, 3), activation=act, padding='same', name="conv5_1")(x)
    x = Conv2D(512, (3, 3), activation=act, padding='same', name="conv5_2")(x)
    x = Conv2D(512, (3, 3), activation=act, padding='same', name="conv5_3")(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), name='pool5', padding='same')(x)    
    
    x = Conv2D(1024, kernel_size=3, padding='same', name='fc6')(x)
    x = Conv2D(1024, kernel_size=1, padding='same', name='fc7')(x)

    model = Model(inp, x, name='vgg16')

    return model


def PixelLink4s(input_shape = None, act = 'relu'):
    vgg = VGG16(input_shape, act)
    fc7 = vgg.get_layer('fc7').output
    conv5_3 = vgg.get_layer('conv5_3').output
    conv4_3 = vgg.get_layer('conv4_3').output
    conv3_3 = vgg.get_layer('conv3_3').output
    
    x = Add()([Conv2D(2,kernel_size=1)(conv5_3), Conv2D(2,kernel_size=1)(fc7)])
    y = Add()([Conv2D(16,kernel_size=1)(conv5_3), Conv2D(16,kernel_size=1)(fc7)])
    
    x = Lambda(upsample)(x)
    y = Lambda(upsample)(y)
    
    x = Add()([x, Conv2D(2,kernel_size=1)(conv4_3)])
    y = Add()([y, Conv2D(16,kernel_size=1)(conv4_3)])
    
    x = Lambda(upsample)(x)
    y = Lambda(upsample)(y)

    x = Add()([x, Conv2D(2,kernel_size=1)(conv3_3)])
    y = Add()([y, Conv2D(16,kernel_size=1)(conv3_3)])

    return Model(vgg.input, [x, y], name = 'pixellink')


def PixelLink2s(input_shape = None, act = 'relu'):
    vgg = VGG16(input_shape, act)
    fc7 = vgg.get_layer('fc7').output
    conv5_3 = vgg.get_layer('conv5_3').output
    conv4_3 = vgg.get_layer('conv4_3').output
    conv3_3 = vgg.get_layer('conv3_3').output
    conv2_2 = vgg.get_layer('conv2_2').output
    
    fc7 = Conv2D(2,kernel_size=1)(fc7)
    conv5_3 = Conv2D(2,kernel_size=1)(conv5_3)
    
    x = Add()([Conv2D(2,kernel_size=1)(fc7), Conv2D(2,kernel_size=1)(conv5_3)])
    y = Add()([Conv2D(16,kernel_size=1)(fc7), Conv2D(16,kernel_size=1)(conv5_3)])
    
    x = Lambda(upsample)(x)
    y = Lambda(upsample)(y)
    
    x = Add()([x, Conv2D(2,kernel_size=1)(conv4_3)])
    y = Add()([y, Conv2D(16,kernel_size=1)(conv4_3)])
    
    x = Lambda(upsample)(x)
    y = Lambda(upsample)(y)

    x = Add()([x, Conv2D(2,kernel_size=1)(conv3_3)])
    y = Add()([y, Conv2D(16,kernel_size=1)(conv3_3)])

    x = Lambda(upsample)(x)
    y = Lambda(upsample)(y)

    x = Add()([x, Conv2D(2,kernel_size=1)(conv2_2)])
    y = Add()([y, Conv2D(16,kernel_size=1)(conv2_2)])
    
    return Model(vgg.input, [x, y], name = 'pixellink')


def upsample(x):
    return tf.image.resize_bilinear(x, size=[K.shape(x)[1]*2, K.shape(x)[2]*2])
