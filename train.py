#!/usr/bin/python
from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Dropout
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
import keras
import tensorflow as tf
import random
import os
import time
import datetime

from data import load_train_data, load_valid_data, load_sliced_data, load_data_including_empty, load_data_including_empty_all
from augment import trainGenerator, validGenerator, trainGenerator2, validGenerator2

# img_rows = 512
# # # img_cols = 500
# img_cols = 512
# img_rows = 128
# img_cols = 496
img_rows = 496
img_cols = 128
batch_size = 50

smooth = 1.
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid = []
        self.lastiter = 0

    def on_epoch_end(self, batch, logs={}):
        self.lastiter += 1
        # with open("/data/pepple/runs/%s/train.txt" % timestamp, "a") as fout:
        with open("./runs/%s/train.txt" % timestamp, "a") as fout:
            for metric in ["loss"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))

        # with open("/data/pepple/runs/%s/valid.txt" % timestamp, "a") as fout:
        with open("./runs/%s/valid.txt" % timestamp, "a") as fout:
            for metric in ["val_loss"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))

    def on_batch_end(self, batch, logs={}):
        self.lastiter += 1
        # with open("/data/pepple/runs/%s/train.txt" % timestamp, "a") as fout:
        with open("./runs/%s/train.txt" % timestamp, "a") as fout:
            for metric in ["loss"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def mean_squared_error_weighted(y_true, y_pred):
    return K.mean(K.square(y_true) * K.square(y_pred - y_true))


def dice_coef(y_true, y_pred):
    # y_pred = K.clip(y_true, 0.0, 1.0)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat(0, [shape[:1] // parts, shape[1:]])
        stride = tf.concat(0, [shape[:1] // parts, shape[1:] * 0])
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    if gpu_count == 1:
        return Model(input=model.inputs, output=outputs)
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init="he_normal")(inputs)
    print('conv1 a', conv1._keras_shape)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv1)
    print('conv1 b', conv1._keras_shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print('pool1', pool1._keras_shape)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init="he_normal")(pool1)
    print('conv2 a', conv2._keras_shape)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv2)
    print('conv2 b', conv2._keras_shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print('pool2', pool2._keras_shape)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init="he_normal")(pool2)
    print('conv3 a', conv3._keras_shape)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv3)
    print('conv3 b', conv3._keras_shape)
    pool3 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(conv3)   # NB beast on keras 1.2
    print('pool3', pool3._keras_shape)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init="he_normal")(pool3)
    print('conv4 a', conv4._keras_shape)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv4)
    print('conv4 b', conv4._keras_shape)
    pool4 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(conv4)
    print('pool4', pool4._keras_shape)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init="he_normal")(pool4)
    print('conv5 a', conv5._keras_shape)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv5)
    print('conv5 b', conv5._keras_shape)


    up6 = UpSampling2D(size=(2, 2))(conv5)
    print('up6', up6._keras_shape)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    print('merge', conv5._keras_shape, conv4._keras_shape, up6._keras_shape)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init="he_normal")(up6)
    print('conv6 a', conv6._keras_shape)
    conv6 = Dropout(0.3)(conv6)
    print('conv6 dropout 1', conv6._keras_shape)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv6)
    print('conv6 b', conv6._keras_shape)
    conv6 = Dropout(0.3)(conv6)
    print('conv6 dropout 2', conv6._keras_shape)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    print('up7', up7._keras_shape)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init="he_normal")(up7)
    print('conv7 a', conv7._keras_shape)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv7)
    print('conv7 b', conv7._keras_shape)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    print('up8', up8._keras_shape)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init="he_normal")(up8)
    print('conv8 a', conv8._keras_shape)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv8)
    print('conv8 b', conv8._keras_shape)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    print('up9', up9._keras_shape)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init="he_normal")(up9)
    print('conv9 a', conv9._keras_shape)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init="he_normal")(conv9)
    print('conv9 b', conv9._keras_shape)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', init="he_normal")(conv9)
    print('conv10', conv10._keras_shape)

    model = Model(input=inputs, output=conv10)
    model.summary()
    return model


def _create_weights_folder():
    # os.mkdir("/data/pepple/runs/%s/" % timestamp)
    # os.mkdir("/data/pepple/runs/%s/weights" % timestamp)
    os.mkdir("./runs/%s/" % timestamp)
    os.mkdir("./runs/%s/weights" % timestamp)


def train_and_predict():
    _create_weights_folder()
    model = get_unet()
    # with open("/data/mri/runs/%s/model.yaml" % timestamp, "w") as fout:
    #     fout.write(model.to_yaml())
    lr = 1e-7
    # model = make_parallel(model, 8)
    # model.compile(optimizer=Adam(lr=lr), loss=mean_squared_error_weighted, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_sliced_data()    # train vs validation split in generator

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    # with open("/data/pepple/runs/%s/params.txt" % timestamp, "w") as fout:
    with open("./runs/%s/params.txt" % timestamp, "w") as fout:
        fout.write("mean\t%.9f\n" % mean)
        fout.write("std\t%.9f\n" % std)
        fout.write("lr\t%.9f\n" % lr)

    imgs_train -= mean
    imgs_train /= std
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    # filepath = "/data/pepple/runs/%s/weights/weights-improvement-{epoch:03d}-{val_loss:.8f}.hdf5" % timestamp
    filepath = "./runs/%s/weights/weights-improvement-{epoch:03d}-{val_loss:.8f}.hdf5" % timestamp
    checkpoint = ModelCheckpoint(filepath)

    history = LossHistory()

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit_generator(trainGenerator(batch_size=batch_size, data=imgs_train, mask=imgs_mask_train), nb_worker=1,
                        validation_data=validGenerator(batch_size=batch_size, data=imgs_train, mask=imgs_mask_train),
                        samples_per_epoch=3000, nb_epoch=500, verbose=1, nb_val_samples=2500,
                        callbacks=[history, checkpoint])  # 3600384

    # model.fit(imgs_train, imgs_mask_train, validation_split=0.4, batch_size=10,
    #           nb_epoch=10000, verbose=1, callbacks=[history, checkpoint])
    # model.fit(imgs_train, imgs_mask_train, validation_data=(imgs_valid, imgs_mask_valid), batch_size=100,
    #           nb_epoch=10000, verbose=1, callbacks=[history, checkpoint])
    return


# just copied train_and_predict but with different datasets (now includes empty segmentations)
def train_and_predict2():
    _create_weights_folder()

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()
    # with open("/data/mri/runs/%s/model.yaml" % timestamp, "w") as fout:
    #     fout.write(model.to_yaml())
    lr = 1e-5
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    # seg_sliced, seg_masks_sliced, empty_seg_sliced, empty_masks_sliced = load_data_including_empty()
    seg_sliced, seg_masks_sliced, empty_seg_sliced, empty_masks_sliced = load_data_including_empty_all()
    print(seg_sliced.shape, seg_masks_sliced.shape, empty_seg_sliced.shape, empty_masks_sliced.shape)

    seg_sliced = seg_sliced.astype('float32')
    seg_masks_sliced = seg_masks_sliced.astype('float32')
    empty_seg_sliced = empty_seg_sliced.astype('float32')
    empty_masks_sliced = empty_masks_sliced.astype('float32')

    mean = np.mean(seg_sliced)  # mean for data centering
    std = np.std(seg_sliced)  # std for data normalization

    with open("./runs/%s/params.txt" % timestamp, "w") as fout:
        fout.write("mean\t%.9f\n" % mean)
        fout.write("std\t%.9f\n" % std)
        fout.write("lr\t%.9f\n" % lr)

    # seg_masks_sliced -= mean
    # seg_masks_sliced /= std
    seg_masks_sliced /= 255.  # scale masks to [0, 1]
    #
    # empty_seg_sliced -= mean
    # empty_seg_sliced /= std
    empty_masks_sliced /= 255.  # scale masks to [0, 1]

    filepath = "./runs/%s/weights/weights-improvement-{epoch:03d}-{val_loss:.8f}.hdf5" % timestamp
    checkpoint = ModelCheckpoint(filepath)

    history = LossHistory()

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit_generator(trainGenerator2(batch_size=batch_size, data=seg_sliced, masks=seg_masks_sliced,
                                        empty_data=empty_seg_sliced, empty_masks=empty_masks_sliced), nb_worker=1,
                        validation_data=validGenerator2(batch_size=batch_size, data=seg_sliced, masks=seg_masks_sliced,
                                        empty_data=empty_seg_sliced, empty_masks=empty_masks_sliced),
                        samples_per_epoch=10000, nb_epoch=500, verbose=1, nb_val_samples=2500,
                        callbacks=[history, checkpoint])
    return


if __name__ == '__main__':
    # 103=9888/96 seg images (no pepple); 87=8352/96 (empty)
    # seg_sliced, seg_masks_sliced, empty_seg_sliced, empty_masks_sliced = load_data_including_empty()

    # train_and_predict()
    train_and_predict2()