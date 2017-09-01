#!/usr/bin/python
from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, merge, \
    GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.losses import categorical_crossentropy
import tensorflow as tf
import os
import time
import datetime

from qlearning_data import load_qtrain_data, load_qvalid_data

img_rows = 64
img_cols = 64
batch_size = 10
smooth = 1.

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid = []
        self.lastiter = 0

    def on_epoch_end(self, batch, logs={}):
        self.lastiter += 1
        with open("./runs/%s/train.txt" % timestamp, "a") as fout:
            for metric in ["loss", "acc"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))

        with open("./runs/%s/valid.txt" % timestamp, "a") as fout:
            for metric in ["val_loss", "val_acc"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))

    def on_batch_end(self, batch, logs={}):
        self.lastiter += 1
        with open("./runs/%s/train.txt" % timestamp, "a") as fout:
            for metric in ["loss"]:
                fout.write("train\t%d\t%s\t%.6f\n" % (self.lastiter, metric, logs.get(metric)))


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


def simple_vgg(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=8):
    inputs = Input((img_rows, img_cols, 2))

    # Block 1
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Classification block
    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    # Create model.
    model = Model(inputs, x, name='vgg_simple')
    return model


def _create_weights_folder():
    os.mkdir("./runs/%s/" % timestamp)
    os.mkdir("./runs/%s/weights" % timestamp)


def train_and_predict():
    _create_weights_folder()
    model = simple_vgg()
    lr = 1e-5
    model.compile(optimizer=Adam(lr=lr), loss=categorical_crossentropy, metrics=['accuracy'])
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    imgs_train, target_train = load_qtrain_data()
    imgs_train = imgs_train.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    # with open("/data/pepple/runs/%s/params.txt" % timestamp, "w") as fout:
    with open("./runs/%s/params.txt" % timestamp, "w") as fout:
        fout.write("mean\t%.9f\n" % mean)
        fout.write("std\t%.9f\n" % std)
        fout.write("lr\t%.9f\n" % lr)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid, target_valid = load_qvalid_data()
    imgs_valid = imgs_valid.astype('float32')
    imgs_valid -= mean
    imgs_valid /= std

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    filepath = "./runs/%s/weights/weights-improvement-{epoch:03d}-{val_loss:.8f}.hdf5" % timestamp
    checkpoint = ModelCheckpoint(filepath)

    history = LossHistory()

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model.fit(imgs_train, target_train, epochs=100, validation_data=(imgs_valid, target_valid), verbose=1,
              callbacks=[history, checkpoint])
    # model.fit_generator(trainGenerator(batch_size=batch_size, data=imgs_train, mask=imgs_mask_train), nb_worker=1,
    #                     validation_data=validGenerator(batch_size=batch_size, data=imgs_train, mask=imgs_mask_train),
    #                     samples_per_epoch=3000, nb_epoch=500, verbose=1, nb_val_samples=2500, callbacks=[history, checkpoint])  # 3600384


if __name__ == '__main__':
    train_and_predict()