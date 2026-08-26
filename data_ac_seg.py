import os, cv2, json, glob
import numpy as np

from data import read_images, slice_data

from sys import platform
if platform == "linux" or platform == "linux2":
    prefix = '/data/yue/pepple'
elif platform == "win32":
    prefix = 'z:/yue/pepple'

# aliases folders
base_folder = os.path.join(prefix, 'acseg')
img_folder = os.path.join(base_folder, 'segmentations')
empty_folder = os.path.join(base_folder, 'empty_segmentations')
raw_folder = os.path.join(base_folder, 'convert_neighboring')
all_raw_folder = os.path.join(base_folder, 'Inflamed')
npy_folder = os.path.join(base_folder, 'npy')
if not os.path.isdir(npy_folder):
    os.makedirs(npy_folder)

# constants
DOWNSAMPLE_SCALE = 2
TRAIN_IMGS_RAW = 'train_imgs_raw.npy'
TRAIN_IMGS_RESIZED = 'train_imgs_resized.npy'
TRAIN_IMGS_MASK = 'train_imgs_mask.npy'
VALID_IMGS_RAW = 'valid_imgs_raw.npy'
VALID_IMGS_RESIZED = 'valid_imgs_resized.npy'
VALID_IMGS_MASK = 'valid_imgs_mask.npy'
TRAIN_SLICED_IMGS = 'train_sliced_imgs.npy'
TRAIN_SLICED_MASK = 'train_sliced_mask.npy'
VALID_SLICED_IMGS = 'valid_sliced_imgs.npy'
VALID_SLICED_MASK = 'valid_sliced_mask.npy'


def get_data():
    # read images
    masks, img_names = get_images(img_folder, ext='_mask.png', npy_path=os.path.join(npy_folder, 'seg_masks.npy'))
    empty_masks, empty_names = get_images(empty_folder, ext='_mask.png', npy_path=os.path.join(npy_folder, 'empty_masks.npy'))
    raw_imgs, converted_imgs = get_raw_imgs(img_names, raw_folder)
    raw_empties, converted_empties = get_raw_imgs(empty_names, raw_folder, is_empty=True)

    with open(os.path.join(npy_folder, 'img_names.txt'), 'w') as fout:
        for idx, img_name in enumerate(img_names):
            fout.write('{},{}\n'.format(img_name.replace('_mask.png', ''), img_name.replace('.png', '')))
    fout.close()
    with open(os.path.join(npy_folder, 'img_names_empty.txt'), 'w') as fout:
        for idx, img_name in enumerate(empty_names):
            fout.write('{},{}\n'.format(img_name.replace('_mask.png', ''), img_name.replace('.png', '')))
    fout.close()
    return raw_imgs, converted_imgs, masks, img_names, raw_empties, converted_empties, empty_masks, empty_names


def get_images(folder, ext, npy_path=''):
    img_names = [x for x in sorted(os.listdir(folder)) if ext in x]
    if os.path.isfile(npy_path):
        data = np.load(npy_path)
        return data, img_names

    data = []
    for idx, img_name in enumerate(img_names):
        cur_img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        data.append(cur_img)
    data = np.asarray(data)
    np.save(npy_path, data)
    return data, img_names


def get_seg_raw_imgs(folder=raw_folder):
    raw_img_names = [x for x in sorted(os.listdir(folder)) if '.TIFF' in x]
    return raw_img_names


def get_raw_imgs(img_names, main_raw_folder=raw_folder, all_raw_folder=all_raw_folder, ext='TIFF', is_empty=False):
    raw_npy = os.path.join(npy_folder, 'raw_imgs{}.npy'.format('_empty' if is_empty else ''))
    converted_npy = os.path.join(npy_folder, 'imgs_cv2_resized{}.npy'.format('_empty' if is_empty else ''))
    if os.path.isfile(raw_npy) and os.path.isfile(converted_npy):
        raw_data = np.load(raw_npy)
        converted_data = np.load(converted_npy)
        return raw_data, converted_data

    raw_img_names = get_seg_raw_imgs(main_raw_folder)

    raw_data = []
    converted_data = []
    for idx, img_name in enumerate(img_names):
        # img_base = img_name.split('-')[1].replace('.png', '.TIFF')
        transformed_raw_name, img_base = transform_to_raw_name(img_name, ext=ext)
        if transformed_raw_name in raw_img_names:
            raw_img_path = os.path.join(main_raw_folder, transformed_raw_name)
        else:
            raw_img_path = find_raw_img_path(transformed_raw_name, all_raw_folder)
        if raw_img_path is None:
            print('not found', raw_img_path)
        raw_img = cv2.imread(raw_img_path, cv2.IMREAD_GRAYSCALE)
        img_shape = raw_img.shape
        new_width, new_height = int(img_shape[0]/DOWNSAMPLE_SCALE), int(img_shape[1]/DOWNSAMPLE_SCALE)
        resized_img = cv2.resize(raw_img, (new_height, new_width), interpolation=cv2.INTER_CUBIC)
        raw_data.append(raw_img)
        converted_data.append(resized_img)
    raw_data = np.asarray(raw_data)
    converted_data = np.asarray(converted_data)
    np.save(raw_npy, raw_data)
    np.save(converted_npy, converted_data)
    return raw_data, converted_data


def transform_to_raw_name(img_name, ext='TIFF'):
    img_name = img_name.replace('.png', '.{}'.format(ext))
    img_name = img_name.replace('Uninflamed_', '')
    img_name = img_name.replace('Inflamed_', '')

    img_base = img_name.split('-')[1]
    img_toks = img_base.split('_')
    img_num = img_toks[3]
    raw_img_name = '{} ({}).{}'.format('_'.join(img_toks[:3]), img_num, ext)
    return raw_img_name, img_base


def find_raw_img_path(img_name, folder=all_raw_folder):
    img_path = glob.glob('{}/**/{}'.format(folder, img_name), recursive=True)
    if len(img_path)>=1:
        return img_path[0]
    else:
        return None


def create_data(is_train=True):
    if is_train:
        raw_path = TRAIN_IMGS_RAW
        resized_path = TRAIN_IMGS_RESIZED
        mask_path = TRAIN_IMGS_MASK
        save_img_path = os.path.join(npy_folder, TRAIN_SLICED_IMGS)
        save_mask_path = os.path.join(npy_folder, TRAIN_SLICED_MASK)
    else:
        raw_path = VALID_IMGS_RAW
        resized_path = VALID_IMGS_RESIZED
        mask_path = VALID_IMGS_MASK
        save_img_path = os.path.join(npy_folder, VALID_SLICED_IMGS)
        save_mask_path = os.path.join(npy_folder, VALID_SLICED_MASK)

    raw_imgs = np.load(os.path.join(npy_folder, raw_path))
    resized_imgs = np.load(os.path.join(npy_folder, resized_path))
    mask = np.load(os.path.join(npy_folder, mask_path))
    
    slice_data(resized_imgs, mask, True, save_img_path, save_mask_path)
    return


def split_train_valid(raw_imgs, converted_imgs, masks, img_names, raw_empties, converted_empties, empty_masks, empty_names, empty_ratio=0.2):
    num_seg = len(img_names)
    num_empty = len(empty_names)

    img_train = []
    img_valid = []
    for idx, img_name in enumerate(img_names):
        if 'Kathryn' in img_name:   # for test as rough lines
            img_valid.append(img_name)
        else:
            img_train.append(img_name)

    num_train = len(img_train)
    num_empty_use = int(np.ceil(num_train*empty_ratio))
    empty_train = empty_names[:num_empty_use]
    empty_valid = empty_names[num_empty_use:]

    # separate img and masks appropriately
    train_imgs_raw, train_imgs_converted, train_imgs_mask = subset_imgs(img_train, img_names, raw_imgs, converted_imgs, masks)
    valid_imgs_raw, valid_imgs_converted, valid_imgs_mask = subset_imgs(img_valid, img_names, raw_imgs, converted_imgs, masks)
    train_imgs_raw2, train_imgs_converted2, train_imgs_mask2 = subset_imgs(empty_train, empty_names, raw_empties, converted_empties, empty_masks)
    valid_imgs_raw2, valid_imgs_converted2, valid_imgs_mask2 = subset_imgs(empty_valid, empty_names, raw_empties, converted_empties, empty_masks)

    train_imgs_raw = np.array(train_imgs_raw + train_imgs_raw2)
    train_imgs_converted = np.array(train_imgs_converted + train_imgs_converted2)
    train_imgs_mask = np.array(train_imgs_mask + train_imgs_mask2)
    np.save(os.path.join(npy_folder, TRAIN_IMGS_RAW), train_imgs_raw)
    np.save(os.path.join(npy_folder, TRAIN_IMGS_RESIZED), train_imgs_converted)
    np.save(os.path.join(npy_folder, TRAIN_IMGS_MASK), train_imgs_mask)
    valid_imgs_raw = np.array(valid_imgs_raw + valid_imgs_raw2)
    valid_imgs_converted = np.array(valid_imgs_converted + valid_imgs_converted2)
    valid_imgs_mask = np.array(valid_imgs_mask + valid_imgs_mask2)
    np.save(os.path.join(npy_folder, VALID_IMGS_RAW), valid_imgs_raw)
    np.save(os.path.join(npy_folder, VALID_IMGS_RESIZED), valid_imgs_converted)
    np.save(os.path.join(npy_folder, VALID_IMGS_MASK), valid_imgs_mask)

    # now save the file names
    img_train += empty_train
    img_valid += empty_valid
    with open(os.path.join(npy_folder, 'train_img_names.txt'), 'w') as fout:
        for idx, img_name in enumerate(img_train):
            fout.write('{},{}\n'.format(img_name.replace('_mask.png', ''), img_name.replace('.png', '')))
    fout.close()
    with open(os.path.join(npy_folder, 'valid_img_names.txt'), 'w') as fout:
        for idx, img_name in enumerate(img_valid):
            fout.write('{},{}\n'.format(img_name.replace('_mask.png', ''), img_name.replace('.png', '')))
    fout.close()

    return img_train, img_valid


def subset_imgs(img_names_sub, all_img_names, imgs_raw, imgs_converted, imgs_mask):
    imgs_raw_sub = []
    imgs_converted_sub = []
    imgs_mask_sub = []
    for img_name in img_names_sub:
        img_idx = all_img_names.index(img_name)
        imgs_raw_sub.append(imgs_raw[img_idx, ])
        imgs_converted_sub.append(imgs_converted[img_idx, ])
        imgs_mask_sub.append(imgs_mask[img_idx, ])
    return imgs_raw_sub, imgs_converted_sub, imgs_mask_sub


def load_data(is_train=True, normalize=True):
    if is_train:
        data_path = TRAIN_SLICED_IMGS
        mask_path = TRAIN_SLICED_MASK
    else:
        data_path = VALID_SLICED_IMGS
        mask_path = VALID_SLICED_MASK

    img_data = np.load(os.path.join(npy_folder, data_path))
    mask_data = np.load(os.path.join(npy_folder, mask_path))

    # scale imgs individually
    num_data, num_rows, num_cols, _ = img_data.shape
    img_data = img_data.astype(np.float32)
    mask_data = mask_data.astype(np.float32)
    img_mean = np.mean(img_data, axis=(1, 2))   # img_mean - everything is grayscale
    img_std = np.std(img_data, axis=(1, 2))     # img_std
    if normalize:
        for idx in range(num_data):
            img_data[idx] -= img_mean[idx]
            img_data[idx] /= img_std[idx]

            mask_data[idx] /= 255
    else:
        img_data = img_data.astype(np.float32)
        mask_data = mask_data.astype(np.float32)

    return img_data, mask_data


if __name__ == '__main__':
    raw_imgs, converted_imgs, masks, img_names, raw_empties, converted_empties, empty_masks, empty_names = get_data()
    # raw_imgs = np.load(os.path.join(npy_folder, 'raw_imgs.npy'))
    # raw_imgs_empty = np.load(os.path.join(npy_folder, 'raw_imgs_empty.npy'))
    # seg_masks = np.load(os.path.join(npy_folder, 'seg_masks.npy'))
    # seg_masks_empty = np.load(os.path.join(npy_folder, 'empty_masks.npy'))

    split_train_valid(raw_imgs, converted_imgs, masks, img_names, raw_empties, converted_empties, empty_masks, empty_names)
    create_data(is_train=True)
    create_data(is_train=False)

    # try load script
    load_data(is_train=True, normalize=True)