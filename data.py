#!/usr/bin/python
from __future__ import print_function

import os
import numpy as np

import cv2
import math
from make_bbox_data import downsample

from sys import platform
if platform == "linux" or platform == "linux2":
    base_path = '/data/pepple/acseg'
    empty_data_path = '/data/pepple/acseg/empty_segmentations'
    data_path = '/data/pepple/acseg/segmentations'
elif platform == "win32":
    base_path = './acseg'
    empty_data_path = './acseg/empty_segmentations'
    data_path = './acseg/segmentations'
npy_folder = './npy'

INCLUDE_KATHRYN = True
MAKE_UNIQUE = True
# save paths
data_type = ''
if not INCLUDE_KATHRYN:
    data_type = '_no_pepple'

npy_path = os.path.join(npy_folder,'{}{}.npy'.format('multi_validb', data_type))
npy_mask_path = os.path.join(npy_folder,'{}{}.npy'.format('multi_mask_validb', data_type))

data_type2 = ''
if MAKE_UNIQUE:
    data_type2 = '_unique'
npy_unique_img_path = os.path.join(npy_folder,'{}.npy'.format('unique_images'))
npy_unique_mask_path = os.path.join(npy_folder,'{}.npy'.format('unique_masks'))
npy_sliced_path = os.path.join(npy_folder,'{}{}.npy'.format('sliced_vertical', data_type2))
npy_sliced_mask_path = os.path.join(npy_folder,'{}{}.npy'.format('sliced_mask_vertical', data_type2))

npy_empty_segs = './npy/empty_segs.npy'
npy_empty_masks = './npy/empty_masks.npy'
npy_empty_segs_sliced = './npy/empty_segs_sliced.npy'
npy_empty_masks_sliced = './npy/empty_masks_sliced.npy'

# new paths
npy_segs_no_pepple = './npy/segmented_no_pepple.npy'     # no pepple
npy_masks_no_pepple = './npy/masks_no_pepple.npy'     # no pepple
npy_segs_no_pepple_sliced = './npy/segmented_no_pepple_sliced.npy'     # no pepple
npy_masks_no_pepple_sliced = './npy/masks_no_pepple_sliced.npy'     # no pepple

image_rows = 512
image_cols = 500
maxtotmod = 25*8


# read images
def read_images(folder=data_path, read_empty=False):
    images = [x for x in os.listdir(folder) if '.png' in x.lower()]
    total = len(images) / 2
    print('total valid images', total)
    total_valid = int(total)

    if not read_empty and os.path.isfile(npy_path) and os.path.isfile(npy_mask_path):
        imgs = np.load(npy_path)
        imgs_mask = np.load(npy_mask_path)
        real_names = [x for x in images if 'mask' not in x]
        return imgs, imgs_mask, real_names
    elif read_empty and os.path.isfile(npy_empty_segs) and os.path.isfile(npy_empty_masks):
        imgs = np.load(npy_empty_segs)
        imgs_mask = np.load(npy_empty_masks)
        real_names = [x for x in images if 'mask' not in x]
        return imgs, imgs_mask, real_names

    imgs = np.ndarray((total_valid, image_rows, image_rows, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((total_valid, image_rows, image_rows, 1), dtype=np.uint8)
    zero_pad = np.zeros((image_rows, image_rows - image_cols, 1), dtype=np.uint8)

    print('-'*30)
    print('Creating validing images...')
    print('-'*30)
    i = 0
    real_names = []
    for idx, image_name in enumerate(images):
        if 'mask' in image_name:
            continue
        print(i, image_name)
        real_names.append(image_name)
        image_mask_name = image_name.split('.')[0] + '_mask.png'

        img = cv2.imread(os.path.join(folder, image_name), cv2.IMREAD_GRAYSCALE)
        img = img.reshape((image_rows, image_cols, 1))
        img = np.concatenate((img, zero_pad), axis=1)   # HACK - add padding at bottom of image for U-Net max pooling

        img_mask = cv2.imread(os.path.join(folder, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = img_mask.reshape((image_rows, image_cols, 1))
        img_mask = np.concatenate((img_mask, zero_pad), axis=1) # HACK - add padding at bottom of image for U-Net max pooling

        imgs[i, ] = img
        imgs_mask[i, ] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i +=1
    print('Loading done.')

    return imgs, imgs_mask, real_names


# break into training and validation data and save
def create_data():
    seg_data, seg_masks, seg_img_names = read_images(folder=data_path)

    # separate nick and kathrynn seg to prevent same image peeking
    # use nick's data for now as that is more carefully segmented - technically non-pepple since ayl and ywu included
    nick_segs_idx = []
    for idx, image_name in enumerate(seg_img_names):
        if 'Kathryn' in image_name:     # pepple filter
            continue
        nick_segs_idx.append(idx)
    nick_seg_data = seg_data[nick_segs_idx,]
    nick_seg_masks = seg_masks[nick_segs_idx,]
    # # save complete non-pepple seg for future reference
    np.save(npy_segs_no_pepple, nick_seg_data)
    np.save(npy_masks_no_pepple, nick_seg_masks)

    # train_val_split = 0.8
    # num_nick_segs, _, _, _ = nick_seg_data.shape
    # num_nick_train = int(num_nick_segs*.8)
    # nick_seg_train = nick_seg_data[:num_nick_train, ]
    # nick_masks_train = nick_seg_masks[:num_nick_train, ]
    # nick_seg_valid = nick_seg_data[num_nick_train:, ]
    # nick_masks_valid = nick_seg_masks[num_nick_train:, ]
    # # np.save(npy_segs_train_no_pepple, nick_seg_train)
    # # np.save(npy_masks_train_no_pepple, nick_masks_train)
    # # np.save(npy_segs_valid_no_pepple, nick_seg_valid)
    # # np.save(npy_masks_valid_no_pepple, nick_masks_valid)

    # doesnt matter for empty segmentations
    empty_data, empty_masks, empty_img_names = read_images(folder=empty_data_path)
    # save complete empty for future reference
    np.save(npy_empty_segs, empty_data)
    np.save(npy_empty_masks, empty_masks)

    # slice data and store
    seg_sliced, seg_masks_sliced = slice_data(nick_seg_data, nick_seg_masks, save_data=False)
    empty_seg_sliced, empty_masks_sliced = slice_data(empty_data, empty_masks, save_data=False)
    np.save(npy_segs_no_pepple_sliced, seg_sliced)
    np.save(npy_masks_no_pepple_sliced, seg_masks_sliced)
    np.save(npy_empty_segs_sliced, empty_seg_sliced)
    np.save(npy_empty_masks_sliced, empty_masks_sliced)

    # # separate into training vs validation
    # num_empty, _, _, _ = empty_data.shape
    # # FIXME - limit empty to 16% of training data
    # num_empty_train = int(min(num_empty * .8, .2*num_nick_train))
    # empty_segs_train = empty_data[:num_empty_train,]
    # empty_masks_train = empty_masks[:num_empty_train,]
    # empty_segs_valid = empty_data[num_empty_train:,]
    # empty_masks_valid = empty_masks[num_empty_train:,]

    # # combine into training and validation
    # seg_train = np.concatenate((nick_seg_train, empty_segs_train), axis=0)
    # masks_train = np.concatenate((nick_masks_train, empty_masks_train), axis=0)
    # seg_valid = np.concatenate((nick_seg_valid, empty_segs_valid), axis=0)
    # masks_valid = np.concatenate((nick_masks_valid, empty_masks_valid), axis=0)
    # np.save(npy_train_segs, seg_train)
    # np.save(npy_train_masks, masks_train)
    # np.save(npy_valid_segs, seg_valid)
    # np.save(npy_valid_masks, masks_valid)

    # # slice data
    # seg_train_sliced, masks_train_sliced = slice_data(seg_train, masks_train, save_data=False)
    # seg_valid_sliced, masks_valid_sliced = slice_data(seg_valid, masks_valid, save_data=False)
    # np.save(npy_train_segs_sliced, seg_train_sliced)
    # np.save(npy_train_masks_sliced, masks_train_sliced)
    # np.save(npy_valid_segs_sliced, seg_valid_sliced)
    # np.save(npy_valid_masks_sliced, masks_valid_sliced)
    return 1


# def load_data_including_empty_sliced():
#     seg_train = np.load(npy_train_segs_sliced)
#     masks_train = np.load(npy_train_masks_sliced)
#     seg_valid = np.load(npy_valid_segs_sliced)
#     masks_valid = np.load(npy_valid_masks_sliced)
#
#     return seg_train, masks_train, seg_valid, masks_valid
#
#
# def load_data_including_empty_orig():
#     seg_train = np.load(npy_train_segs)
#     masks_train = np.load(npy_train_masks)
#     seg_valid = np.load(npy_valid_segs)
#     masks_valid = np.load(npy_valid_masks)
#     return seg_train, masks_train, seg_valid, masks_valid
#
#
# def load_data_including_empty():
#     seg_train, masks_train, seg_valid, masks_valid = load_data_including_empty_orig()
#     seg_train_sliced, masks_train_sliced, seg_valid_sliced, masks_valid_sliced = load_data_including_empty_sliced()
#     return seg_train, masks_train, seg_valid, masks_valid, seg_train_sliced, masks_train_sliced, seg_valid_sliced, \
#            masks_valid_sliced
def load_data_including_empty():
    seg_sliced = np.load(npy_segs_no_pepple_sliced)
    seg_masks_sliced = np.load(npy_masks_no_pepple_sliced)
    empty_seg_sliced = np.load(npy_empty_segs_sliced)
    empty_masks_sliced = np.load(npy_empty_masks_sliced)
    return seg_sliced, seg_masks_sliced, empty_seg_sliced, empty_masks_sliced


def load_original_data_including_empty():
    seg = np.load(npy_segs_no_pepple)
    seg_masks = np.load(npy_masks_no_pepple)
    empty_seg = np.load(npy_empty_segs)
    empty_masks = np.load(npy_empty_masks)
    return seg, seg_masks, empty_seg, empty_masks


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
        # if 'Kathryn' in image_name:     # pepple filter
        #     continue
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

    read_images(folder=data_path)
    print('saving train size', imgs.shape, imgs_mask.shape)
    # np.save(npy_path, imgs)
    # np.save(npy_mask_path, imgs_mask)
    np.save(npy_path, imgs[real_data_indices,:])
    np.save(npy_mask_path, imgs_mask[real_data_indices,:])
    print('Saving to .npy files done.')


def load_valid_data():
    imgs_valid = np.load(npy_path)
    imgs_mask_valid = np.load(npy_mask_path)
    print('valid size', imgs_valid.shape, imgs_mask_valid.shape)
    return imgs_valid, imgs_mask_valid


def create_train_data():
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
    # cropped_rows = 128
    # cropped_cols = 496
    cropped_rows = 496
    cropped_cols = 128
    pixel_overlap = 8

    data_shape = data.shape
    mask_shape = mask.shape
    img_rows = data_shape[1]
    img_cols = data_shape[2]
    # img_rows = 512
    # img_cols = 512

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


# include kathryhn's data
def load_data_including_empty_all():
    seg_sliced = np.load(npy_sliced_path)
    seg_masks_sliced = np.load(npy_sliced_mask_path)
    empty_seg_sliced = np.load(npy_empty_segs_sliced)
    empty_masks_sliced = np.load(npy_empty_masks_sliced)
    return seg_sliced, seg_masks_sliced, empty_seg_sliced, empty_masks_sliced


# break into training and validation data and save
def create_data_all():
    seg_data, seg_masks, seg_img_names = read_images(folder=data_path)  # original images

    if MAKE_UNIQUE:     # remove duplicate segmentations to avoid confusion
        seg_bases = []
        seg_base_idx = []
        seg_unique_img_names = []
        for idx, seg_name in enumerate(seg_img_names):
            seg_base = '_'.join(seg_name.split('-')[1:])
            if seg_base not in seg_bases:   # since sorted alpha and kathryn after nick
                seg_bases.append(seg_base)
                seg_base_idx.append(idx)
                seg_unique_img_names.append(seg_name)
        seg_data = seg_data[seg_base_idx, ]
        seg_masks = seg_masks[seg_base_idx, ]
        # seg_img_names = seg_img_names[seg_base_idx]
        with open(os.path.join(npy_folder, 'unique_img_names.txt'), 'w') as fout:
            for uname in seg_unique_img_names:
                fout.write('{}\n'.format(uname))
        np.save(npy_unique_img_path, seg_data)
        np.save(npy_unique_mask_path, seg_masks)

    # doesnt matter for empty segmentations
    empty_data, empty_masks, empty_img_names = read_images(folder=empty_data_path, read_empty=True)
    # save complete empty for future reference
    if not os.path.isfile(npy_empty_segs):
        np.save(npy_empty_segs, empty_data)
    if not os.path.isfile(npy_empty_masks):
        np.save(npy_empty_masks, empty_masks)

    # slice data and store
    seg_sliced, seg_masks_sliced = slice_data(seg_data, seg_masks, save_data=False)
    empty_seg_sliced, empty_masks_sliced = slice_data(empty_data, empty_masks, save_data=False)
    np.save(npy_sliced_path, seg_sliced)
    np.save(npy_sliced_mask_path, seg_masks_sliced)
    np.save(npy_empty_segs_sliced, empty_seg_sliced)
    np.save(npy_empty_masks_sliced, empty_masks_sliced)
    return 1


if __name__ == '__main__':
        # create_train_data()
        # create_valid_data()

        # imgs_train, imgs_mask_train = load_train_data()
        # imgs_valid, imgs_mask_valid = load_valid_data()
        # slice_data(imgs_train, imgs_mask_train)
        # load_sliced_data()

        # new training data with empty segmentations
        # create_data()
        # load_data_including_empty()
        create_data_all()

        # # test generator
        # from augment import validGenerator2
        # data, masks, empty_data, empty_masks = load_data_including_empty()
        # validGenerator2(32, data, masks, empty_data, empty_masks)