#!/usr/bin/python
from __future__ import print_function

import os
import numpy as np

import cv2
import math

#data_path = '/data/pepple/acseg'
# data_path = '/data/pepple/acseg/empty_segmentations'
data_path = '/data/pepple/acseg/segmentations'
# data_path = './acseg/segmentations'

npy_path = './npy/multi_validb.npy'     # includes pepple
npy_mask_path = './npy/multi_mask_validb.npy'
npy_path = './npy/multi_validb_no_pepple.npy'
npy_mask_path = './npy/multi_mask_validb_no_pepple.npy'
# npy_sliced_path = './npy/sliced_data_no_pepple.npy'
# npy_sliced_mask_path = './npy/sliced_mask_no_pepple.npy'
npy_sliced_path = './npy/sliced_vertical_no_pepple.npy'
npy_sliced_mask_path = './npy/sliced_mask_vertical_no_pepple.npy'


image_rows = 512
image_cols = 500

maxtotmod = 25*8


# FIXME - only take nicks data, vs nick+kathrynn, vs kathrynn
# careful when using same image from nick and kathrynn
def create_valid_data():
    # valid_data_path = os.path.join(data_path, 'valid')
    valid_data_path = data_path
    images = os.listdir(valid_data_path)
    total = len(images) / 2
    print('total valid images', total)
    total_valid = int(total)
    # total = int(math.floor(1.0 * total / maxtotmod) * maxtotmod)

    imgs = np.ndarray((total_valid, image_rows, image_rows, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((total_valid, image_rows, image_rows, 1), dtype=np.uint8)
    zero_pad = np.zeros((image_rows, image_rows-image_cols, 1), dtype=np.uint8)
    # zero_pad_mask = np.zeros((image_rows, image_rows-image_cols, 1), dtype=np.uint8)

    real_data_indices = []
    i = 0
    print('-'*30)
    print('Creating validing images...')
    print('-'*30)
    for idx, image_name in enumerate(images):
        if 'Kathryn' in image_name:     # pepple filter
            continue
        if 'mask' in image_name:
            continue
        if i == total:
            break

        print(i, image_name)
        real_data_indices.append(i)
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        # img = cv2.imread(os.path.join(valid_data_path, image_name))
        # # only care about 1 dimension since greyscale
        # img = img[:,:,0]
        img = cv2.imread(os.path.join(valid_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = img.reshape((image_rows, image_cols, 1))
        # HACK - add padding at bottom of image
        img = np.concatenate((img, zero_pad), axis=1)

        img_mask = cv2.imread(os.path.join(valid_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = img_mask.reshape((image_rows, image_cols, 1))
        # HACK - add padding at bottom of image
        img_mask = np.concatenate((img_mask, zero_pad), axis=1)

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    print('saving train size', imgs.shape, imgs_mask.shape)
    # np.save(npy_path, imgs)
    # np.save(npy_mask_path, imgs_mask)
    np.save(npy_path, imgs[real_data_indices,:])
    np.save(npy_mask_path, imgs_mask[real_data_indices,:])
    print('Saving to .npy files done.')


def load_valid_data():
    imgs_valid = np.load(npy_path)
    imgs_mask_valid = np.load(npy_mask_path)
    # imgs_valid = np.load('./npy/imgs_valid.npy')
    # imgs_mask_valid = np.load('./npy/imgs_mask_valid.npy')
    print('valid size', imgs_valid.shape, imgs_mask_valid.shape)
    return imgs_valid, imgs_mask_valid


def create_train_data():
    # train_data_path = os.path.join(data_path, 'train')
    train_data_path = data_path
    images = os.listdir(train_data_path)
    total = len(images) / 2
    print('total train images', total)

    # # get 80% for training
    # total = int(total*.8)
    total = int(total)
    # total = int(math.floor(1.0 * total / maxtotmod) * maxtotmod)

    imgs = np.ndarray((total, image_rows, image_rows, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_rows, 1), dtype=np.uint8)
    zero_pad = np.zeros((image_rows, image_rows-image_cols, 1), dtype=np.uint8)
    # zero_pad_mask = np.zeros((image_rows, image_rows-image_cols, 1), dtype=np.uint8)

    i = 0
    real_data_indices = []
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for idx, image_name in enumerate(images):
        # if idx > total:  # this is now validation data
        #     break

        if 'Kathryn' in image_name:     # pepple filter
            continue
        if 'mask' in image_name:
            continue
        if i == total:
            break

        print(i, image_name)
        real_data_indices.append(i)
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        # img = cv2.imread(os.path.join(train_data_path, image_name))
        # # only care about 1 dimension since greyscale
        # img = img[:,:,0]
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img = img.reshape((image_rows, image_cols, 1))
        # HACK - add padding at bottom of image
        img = np.concatenate((img, zero_pad), axis=1)

        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = img_mask.reshape((image_rows, image_cols, 1))
        # HACK - add padding at bottom of image
        img_mask = np.concatenate((img_mask, zero_pad), axis=1)

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    print('saving train size', imgs.shape, imgs_mask.shape)
    # np.save(npy_path, imgs)
    # np.save(npy_mask_path, imgs_mask)
    np.save(npy_path, imgs[real_data_indices,:])
    np.save(npy_mask_path, imgs_mask[real_data_indices,:])
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load(npy_path)
    imgs_mask_train = np.load(npy_mask_path)
    # imgs_train = np.load('./npy/imgs_train.npy')
    # imgs_mask_train = np.load('./npy/imgs_mask_train.npy')
    print('train size', imgs_train.shape, imgs_mask_train.shape)
    return imgs_train, imgs_mask_train


def slice_data(data, mask, save_data=True):
    img_rows = 512
    img_cols = 512
    # cropped_rows = 128
    # cropped_cols = 496
    cropped_rows = 496
    cropped_cols = 128
    pixel_overlap = 8

    data_shape = data.shape
    mask_shape = mask.shape

    num_images = int(data_shape[0]*((img_rows - cropped_rows)/pixel_overlap * (img_cols - cropped_cols)/pixel_overlap))
    sliced_data = np.ndarray((num_images, cropped_rows, cropped_cols, 1), dtype=np.float32)
    sliced_mask = np.ndarray((num_images, cropped_rows, cropped_cols, 1), dtype=np.float32)

    # make it bigger
    counter = 0
    for k in range(0, data_shape[0]):
        for i in range(0, data_shape[1]-cropped_rows, pixel_overlap):
            for j in range(0, data_shape[2]-cropped_cols, pixel_overlap):
                cur_index = k*((img_rows-cropped_rows)/pixel_overlap)*((img_cols-cropped_cols)/pixel_overlap) \
                            + (i/pixel_overlap)*((img_cols-cropped_cols)/pixel_overlap) + j/pixel_overlap
                cur_index = int(cur_index)
                # print(k, i, j, counter, cur_index)
                # counter += 1
                cur_sub_img = data[k, i:i+cropped_rows, j:j+cropped_cols, 0]
                sliced_data[cur_index,:,:,0] = cur_sub_img
                # concatenate is slow
                # cur_sub_img = cur_sub_img.reshape(1, cropped_rows, cropped_cols, 1)
                # sliced_data = np.concatenate((sliced_data, cur_sub_img), axis=0)
                cur_mask = mask[k, i:i+cropped_rows, j:j+cropped_cols, 0]
                # cur_mask = cur_mask.reshape(1, cropped_rows, cropped_cols, 1)
                # sliced_mask = np.concatenate((sliced_mask, cur_mask), axis=0)
                sliced_mask[cur_index,:,:,0] = cur_mask

    print('data:', sliced_data.shape, 'mask:', sliced_mask.shape)

    if save_data:
        np.save(npy_sliced_path, sliced_data)
        np.save(npy_sliced_mask_path, sliced_mask)
    return sliced_data, sliced_mask


def load_sliced_data():
    imgs_train = np.load(npy_sliced_path)
    imgs_mask_train = np.load(npy_sliced_mask_path)
    print('data size', imgs_train.shape, imgs_mask_train.shape)
    return imgs_train, imgs_mask_train


if __name__ == '__main__':
        # create_train_data()
        # create_valid_data()

        # imgs_train, imgs_mask_train = load_train_data()
        # imgs_valid, imgs_mask_valid = load_valid_data()
        # slice_data(imgs_train, imgs_mask_train)
        load_sliced_data()