import tensorflow as tf
from args import args

def VGG(inputs,regularizers=None):
    # blook 1
    conv1 = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(inputs)
    conv1 = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='SAME')(conv1)
    # block 2
    conv2 = tf.keras.layers.Conv2D(128,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(pool1)
    conv2 = tf.keras.layers.Conv2D(128,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='SAME')(conv2)
    # block 3
    conv3 = tf.keras.layers.Conv2D(256,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(pool2)
    conv3 = tf.keras.layers.Conv2D(256,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(conv3)
    conv3 = tf.keras.layers.Conv2D(256,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='SAME')(conv3)
    # block 4
    conv4 = tf.keras.layers.Conv2D(512,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(pool3)
    conv4 = tf.keras.layers.Conv2D(512,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(conv4)
    conv4 = tf.keras.layers.Conv2D(512,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='SAME')(conv4)
    # block 5
    conv5 = tf.keras.layers.Conv2D(512,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(pool4)
    conv5 = tf.keras.layers.Conv2D(512,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(conv5)
    conv5 = tf.keras.layers.Conv2D(512,3,padding='same',activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    pool5 = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='SAME')(conv5)
    return [pool1,pool2,pool3,pool4,pool5]

def DSPP(inputs,channels,rates,regularizers=None):
    dspp = []
    atrous = tf.keras.layers.Conv2D(channels,1,activation='relu',kernel_regularizer=regularizers,bias_regularizer=regularizers)(inputs)
    dspp.append(atrous)
    for rate in rates:
        atrous = tf.keras.layers.Conv2D(
            channels,3,padding='same',dilation_rate=rate,activation='relu',
            kernel_regularizer=regularizers,bias_regularizer=regularizers)(inputs)
        dspp.append(atrous)
    # image pooling
    dims = tf.keras.backend.int_shape(inputs)[1:3]
    pooled = tf.keras.layers.AveragePooling2D(dims)(inputs)
    pooled = tf.keras.layers.Conv2D(256,1,activation='relu')(pooled)
    pooled = tf.keras.layers.UpSampling2D(dims)(pooled)
    dspp.append(pooled)
    # merge
    merge = tf.keras.layers.Concatenate(axis=-1)(dspp)
    merge = ConvBN(merge,channels,3,activation='relu',regularizers=regularizers)
    return merge

def decode_block(inputs,skip,channels,kernels=4,stride=2,unbalance=False,concate=True,regularizers=None):
    tconv = tf.keras.layers.Conv2DTranspose(
        channels,kernels,strides=stride,padding='same',
        kernel_regularizer=regularizers,bias_regularizer=regularizers)(inputs)
    if concate:
        if unbalance:
            skip = tf.keras.layers.Conv2D(channels//4,1)(skip)
        fuse = tf.keras.layers.Concatenate()([tconv,skip])
        fuse = ConvBN(fuse,channels,1,activation='relu',regularizers=regularizers)
        return fuse
    else:
        return tconv

def EUnet(shape):
    inputs = tf.keras.Input(shape=shape)

    # VGG
    pool = VGG(inputs)
    # DSPP
    dspp = DSPP(pool[4],256,[1,3,6])
    # transpose conv layers
    tconv1 = decode_block(dspp,pool[3],512,unbalance=True)
    tconv2 = decode_block(tconv1,pool[2],256,unbalance=True)
    tconv3 = decode_block(tconv2,pool[1],128,unbalance=True)
    tconv4 = decode_block(tconv3,pool[0],64,unbalance=True)
    tconv5 = decode_block(tconv4,inputs,args.categories,4,2,False,False)
    
    model = tf.keras.Model(inputs,tconv5)
    return model

def ConvBN(inputs,filters,kernels,stride=1,padding='same',rate=1,activation=None,regularizers=None):
    conv = tf.keras.layers.Conv2D(
        filters,kernels,stride,
        padding=padding,dilation_rate=rate,activation=activation,
        kernel_regularizer=regularizers,bias_regularizer=regularizers)(inputs)
    bn = tf.keras.layers.BatchNormalization()(conv)
    return bn