from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Convolution2D, UpSampling2D

def build_model():

    img_w, img_h = 256, 256
    kernel = 3

    #Encoder (VGG16 pre-trained: imagenet)
    base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (img_w,img_h,3))
    x = UpSampling2D()(base_model.output)

    #x = Dropout(0.5)(x)

    #Decoder (Scratch)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)

    #x = Dropout(0.5)(x)

    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(512, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)

    #x = Dropout(0.5)(x)

    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(256, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)

    #x = Dropout(0.5)(x)

    x = Convolution2D(128, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(128, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = UpSampling2D()(x)

    #x = Dropout(0.5)(x)

    x = Convolution2D(64, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution2D(64, kernel, kernel, border_mode='same')(x)
    x = Activation('relu')(x)

    #x = Dropout(0.5)(x)

    x = Convolution2D(1, 1, 1, border_mode='same')(x)
    x = Activation('sigmoid')(x)

    head_model = Model(input = base_model.input, output = x)
    head_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    for layer in base_model.layers:
        layer.trainable = False

    return head_model
