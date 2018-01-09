import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
import subprocess

from sys import platform
if platform == "linux" or platform == "linux2":
    seg_path = '/data/pepple/acseg/segmentations'

    base_folder = '/home/yue/pepple/acseg'
    # re-ran vertical transform (lr=1e-5)
    weights_base = '/home/yue/pepple/runs/2017-11-09-23-47-29'
    # test_weights = 'weights-improvement-025--0.96921231.hdf5'
    # test_weights = 'weights-improvement-050--0.97040034.hdf5'
    test_weights = 'weights-improvement-100--0.96851523.hdf5'

    # with empty seg training data
    weights_base = '/home/yue/pepple/runs/2017-11-09-10-26-19'
    # test_weights = 'weights-improvement-025--0.95883078.hdf5'
    # test_weights = 'weights-improvement-050--0.95777710.hdf5'
    test_weights = 'weights-improvement-100--0.96128662.hdf5'

    weights_base = './runs/2017-12-10-14-57-11'
    test_weights = 'weights-improvement-250--0.96331053.hdf5'

    # more balanced training data with more empty segmentations
    weights_base = './runs/2017-12-11-16-23-19'
    test_weights = 'weights-improvement-100--0.95683613.hdf5'

    # more balanced training data with more empty segmentations and no random shift in intensities
    weights_base = './runs/2017-12-11-23-28-26'
    test_weights = 'weights-improvement-100--0.96009621.hdf5'

    # includes more (unique) images
    weights_base = './runs/2017-12-12-17-00-53'
    test_weights = 'weights-improvement-040--0.95886530.hdf5'
    # test_weights = 'weights-improvement-100--0.80790919.hdf5'
    # test_weights = 'weights-improvement-129--0.88892231.hdf5'
    # test_weights = 'weights-improvement-174--0.83855137.hdf5'
    test_weights = 'weights-improvement-256--0.85407762.hdf5'

    # # includes more (unique) images - more range in augmentation
    weights_base = './runs/2017-12-13-17-24-48'
    test_weights = 'weights-improvement-066--0.72392724.hdf5'
    # test_weights = 'weights-improvement-054--0.62857166.hdf5'
    test_weights = 'weights-improvement-205--0.41045597.hdf5'
elif platform == "win32":
    seg_path = './acseg/segmentations'

    base_folder = './acseg'
    weights_base = './runs/runVerticalNew'
    # test_weights = 'weights-improvement-025--0.96921231.hdf5'
    # test_weights = 'weights-improvement-050--0.97040034.hdf5'
    test_weights = 'weights-improvement-100--0.96851523.hdf5'

    weights_base = './runs/runEmptySeg'
    # test_weights = 'weights-improvement-025--0.95883078.hdf5'
    # test_weights = 'weights-improvement-050--0.95777710.hdf5'
    test_weights = 'weights-improvement-100--0.96128662.hdf5'

    weights_base = './runs/runEmptyAug'
    test_weights = 'weights-improvement-250--0.96331053.hdf5'

    # more balanced training data with more empty segmentations
    weights_base = './runs/runEmptyBalanced0'
    test_weights = 'weights-improvement-100--0.95683613.hdf5'

    # more balanced training data with more empty segmentations and no random shift in intensities
    weights_base = './runs/runEmptyBalanced'
    test_weights = 'weights-improvement-100--0.96009621.hdf5'

    # includes more (unique) images
    weights_base = './runs/runAllUnique'
    test_weights = 'weights-improvement-040--0.95886530.hdf5'
    # test_weights = 'weights-improvement-100--0.80790919.hdf5'
    # test_weights = 'weights-improvement-129--0.88892231.hdf5'
    # test_weights = 'weights-improvement-174--0.83855137.hdf5'
    test_weights = 'weights-improvement-256--0.85407762.hdf5'

    # # includes more (unique) images - more range in augmentation
    # weights_base = './runs/runAllUnique2'
    # test_weights = 'weights-improvement-066--0.72392724.hdf5'
    # # test_weights = 'weights-improvement-054--0.62857166.hdf5'
    # test_weights = 'weights-improvement-205--0.41045597.hdf5'

weights_folder = os.path.join(weights_base, 'weights')
VOLUME_FOLDER = os.path.join(base_folder, 'Volume_tiffs')
stack_type = 'Uninflamed'
# stack_type = 'Inflamed'
STACK_FOLDER = os.path.join(VOLUME_FOLDER, stack_type)
CONVERTED_FOLDER = os.path.join(base_folder, 'Volume_converted', stack_type)
if not os.path.exists(CONVERTED_FOLDER):
    os.makedirs(CONVERTED_FOLDER)

DATA_RAW_ROWS = 1024
DATA_RAW_COLS = 1000
DATA_TRAINING_ROWS = 512
DATA_TRAINING_COLS = 512
DATA_TRAINING_COLS_RAW = 500
STRIP_ROWS = 496
STRIP_COLS = 128
num_aug = 96

from data import slice_data
# from analyser import predict_image_mask
from analyser import combine_predicted_strips_into_image


def load_params(folder=weights_base):
    params = []
    with open(os.path.join(folder, "params.txt")) as fin:
        for l in fin:
            arr = l.rstrip().split("\t")
            params.append(np.float32(arr[1]))
    return params


def convert_stack(folder=STACK_FOLDER):
    stack_dirs = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    for stack_dir in stack_dirs:
        tiff_dir = os.path.join(folder, stack_dir)
        tiff_out_dir = os.path.join(CONVERTED_FOLDER, stack_dir)
        if not os.path.exists(tiff_out_dir):
            os.makedirs(tiff_out_dir)

        stack_tiffs = [x for x in os.listdir(tiff_dir) if '.tiff' in x.lower()]
        print('stack_tiff', stack_tiffs)

        for tiff_name in stack_tiffs:
            convert_image(tiff_dir, tiff_name, tiff_out_dir)
            # check conversion
            # cv2.imread(os.path.join(tiff_out_dir, tiff_name.replace('.TIFF', '.png')), cv2.IMREAD_GRAYSCALE).shape
    return


def convert_image(tiff_dir, tiff_name, tiff_out_dir, options='-scale 50%'):
    # convert -scale 50% "#{f}" "#{fout}
    image_magick_cmd = '{} {}'.format('convert', options)
    tiff_in_path = os.path.join(tiff_dir, tiff_name)
    tiff_out_path = os.path.join(tiff_out_dir, tiff_name.replace('.TIFF', '.png'))

    shell_arguments = [image_magick_cmd, tiff_in_path, tiff_out_path]
    print(shell_arguments)
    # subprocess.call([image_magick_cmd, tiff_in_path, tiff_out_path])
    subprocess.call(' '.join([image_magick_cmd, '"{}"'.format(tiff_in_path), '"{}"'.format(tiff_out_path)]), shell=True)
    return


def check_converted_stack(folder=CONVERTED_FOLDER):
    stack_dirs = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    for stack_dir in stack_dirs:
        img_dir = os.path.join(folder, stack_dir)
        stack_imgs = [x for x in os.listdir(img_dir) if '.png' in x.lower()]
        orig_dir = os.path.join(STACK_FOLDER, stack_dir)
        for idx, img_name in enumerate(stack_imgs):
            cur_img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_GRAYSCALE)
            if cur_img.shape!= (DATA_TRAINING_ROWS, DATA_TRAINING_COLS_RAW):
                print('{} has wrong shape {}'.format(img_name, cur_img.shape))

            # check mean and std
            orig_img = cv2.imread(os.path.join(orig_dir, img_name.replace('.png', '.TIFF')), cv2.IMREAD_GRAYSCALE)
            print('orig_shape={}; converted_shape={}; orig_mean={}; orig_std={}; c_mean={}; c_std={}'
                  .format(orig_img.shape, cur_img.shape, np.mean(orig_img), np.std(orig_img), np.mean(cur_img), np.std(cur_img)))
    return


def predict_converted_stacks():
    from train import get_unet #, dice_coef, K

    model = get_unet()
    weight_path = '{}/{}'.format(weights_folder, test_weights)
    print('predict_stack; weight_path={}'.format(weight_path))
    model.load_weights(weight_path)
    print('loaded weights')

    stack_dirs = [name for name in os.listdir(CONVERTED_FOLDER) if os.path.isdir(os.path.join(CONVERTED_FOLDER, name))]
    for stack_dir in stack_dirs:
        stack_path = os.path.join(CONVERTED_FOLDER, stack_dir)
        stack_preds = predict_converted_stack(model, folder=stack_path)
    return


def load_stack(stack_folder, img_rows=DATA_RAW_ROWS, img_cols=DATA_RAW_COLS):
    orig_npy = os.path.join(stack_folder, 'data_orig.npy')
    stripped_npy = os.path.join(stack_folder, 'data_stripped.npy')
    stack_img_data = load_stack_orig(stack_folder, img_rows, img_cols)

    num_images = stack_img_data.shape[0]
    if os.path.isfile(stripped_npy):
        strip_data = np.load(stripped_npy)
    else:
        # store strips by image for easy access
        strip_data = np.ndarray((num_images, num_aug, STRIP_ROWS, STRIP_COLS, 1), dtype=np.float32)
        for idx in range(num_images):
            cur_img_data = stack_img_data[idx, ].reshape([1] + list(stack_img_data[idx, ].shape))
            cur_strip, _ = slice_data(cur_img_data, cur_img_data, save_data=False)     # because different save_path
            strip_data[idx, ] = cur_strip
        np.save(stripped_npy, strip_data)

    return stack_img_data, strip_data


def load_stack_orig(stack_folder, img_rows=DATA_RAW_ROWS, img_cols=DATA_RAW_COLS):
    orig_npy = os.path.join(stack_folder, 'data_orig.npy')
    if os.path.isfile(orig_npy):
        stack_img_data = np.load(orig_npy)
        stack_images = []
        with open('{}/img_names.txt'.format(stack_folder), 'r') as fin:
            for l in fin.readlines():
                stack_images.append(l.rstrip())
    else:
        stack_images = sorted(os.listdir(stack_folder))
        stack_images = [x for x in stack_images if '.png' in x]
        # print('stack_images', stack_images)
        with open('{}/img_names.txt'.format(stack_folder), 'w') as fout:
            for img_name in stack_images:
                fout.write("%s\n" % img_name)

        stack_img_data = np.ndarray((len(stack_images), img_rows, DATA_TRAINING_COLS), dtype=np.float32)
        zero_pad = np.zeros((img_rows, DATA_TRAINING_COLS - img_cols), dtype=np.uint8)

        for idx, img_name in enumerate(stack_images):
            print(idx, img_name)
            img_path = os.path.join(stack_folder, img_name)
            cur_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            cur_img = np.concatenate((cur_img, zero_pad), axis=1)  # HACK - add padding at bottom of image for U-Net max pooling
            stack_img_data[idx, ] = cur_img
        stack_img_data = stack_img_data.reshape(list(stack_img_data.shape)+[1])
        np.save(orig_npy, stack_img_data)
    return stack_img_data, stack_images


def predict_converted_stack(model, folder=CONVERTED_FOLDER, visualise=True, do_save=True):
    # stack_img_data, stack_strips = load_stack(stack_folder=folder, img_rows=DATA_TRAINING_ROWS, img_cols=DATA_TRAINING_COLS_RAW)
    stack_img_data, stack_img_names = load_stack_orig(stack_folder=folder, img_rows=DATA_TRAINING_ROWS, img_cols=DATA_TRAINING_COLS_RAW)
    num_images = stack_img_data.shape[0]
    raw_aug = 96
    pred_masks = np.ndarray((num_images, DATA_TRAINING_ROWS, DATA_TRAINING_COLS), dtype=np.float32)

    if do_save:
        plt.switch_backend('agg')
        output_folder = '{}/pred_figs_unique_v2_205'.format(folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    stack_mean, stack_std, lr = load_params()
    # stack_mean = np.mean(stack_strips)
    # stack_std = np.std(stack_strips)
    for idx in range(num_images):
        # slice in place as strip data too big for memory
        cur_img_data = stack_img_data[idx, ].reshape([1] + list(stack_img_data[idx, ].shape))
        cur_strips, _ = slice_data(cur_img_data, cur_img_data, save_data=False)  # because different save_path

        # normalize
        # stack_mean = np.mean(cur_strips)
        # stack_std = np.std(cur_strips)
        # cur_strips = cur_strips.astype('float32')
        # cur_strips -= stack_mean
        # cur_strips /= stack_std

        output = model.predict(cur_strips, batch_size=10)  # inference step
        mask_pred = combine_predicted_strips_into_image(output, 0, num_aug=int(raw_aug), img_rows=DATA_TRAINING_ROWS,
                                                        img_cols=DATA_TRAINING_COLS, debug_mode=False)
        pred_masks[idx, ] = mask_pred

        if visualise:
            plt.figure(1)   # side-by-side
            plt.clf()
            plt.subplot(131)
            plt.imshow(stack_img_data[idx, :, :, 0])
            plt.subplot(132)
            plt.imshow(mask_pred)
            plt.subplot(133)
            plt.imshow(stack_img_data[idx, :, :, 0])
            # non_zeros = np.transpose(np.nonzero(mask_pred))
            pred_threshold = 0.9  # very conservative threshold
            thresh_points = np.argwhere(mask_pred > pred_threshold)  # n*2 where either 1st/2nd coord below pred_threshold
            plt.scatter(x=thresh_points[:, 1], y=thresh_points[:, 0], c='yellow', s=1)
            if do_save:
                cur_name = stack_img_names[idx]
                plt.savefig('{}/pred_{}.png'.format(output_folder, cur_name.replace('.png', '')))

    mask_npy = os.path.join(folder, 'mask_preds.npy')
    np.save(mask_npy, pred_masks)
    return pred_masks


def make_stack_data():
    return


def visualise_preds(folder):
    stack_img_data, seg_img_names = load_stack_orig(stack_folder=folder)
    pred_masks = np.load('{}/mask_preds.npy'.format(folder))

    num_images = stack_img_data.shape[0]
    for idx in range(num_images):
        mask_pred = pred_masks[idx, ]
        plt.figure(1)  # side-by-side
        plt.clf()
        plt.subplot(131)
        plt.imshow(stack_img_data[idx, :, :, 0])
        plt.subplot(132)
        plt.imshow(mask_pred)
        plt.subplot(133)
        plt.imshow(stack_img_data[idx, :, :, 0])
        # non_zeros = np.transpose(np.nonzero(mask_pred))
        pred_threshold = 0.9  # very conservative threshold
        thresh_points = np.argwhere(mask_pred > pred_threshold)  # n*2 where either 1st/2nd coord below pred_threshold
        plt.scatter(x=thresh_points[:, 1], y=thresh_points[:, 0], c='yellow', s=1)

    return


def compare_seg_vs_converted(seg_folder, converted_folder):
    # find image in seg_folder that follows converted_folder format
    stack_names = [name for name in os.listdir(converted_folder) if os.path.isdir(os.path.join(converted_folder, name))]
    '20170710mouse1_Day-7_Left'
    for name in stack_names:
        seg_img_names = sorted(list(glob.glob('{}/*{}*.png'.format(seg_folder, name))))
        for seg_name in seg_img_names:
            if 'mask' in seg_name:
                continue
            seg_toks = seg_name.split('_')
            img_aaron_name = '_'.join(seg_toks[1:])
            # for Uninflamed
            # img_aaron_name = img_aaron_name.replace(seg_toks[2], '{}-{}'.format(seg_toks[2][:-1], seg_toks[2][-1]))
            img_num = seg_toks[-1].replace('.png', '')
            img_converted_name = img_aaron_name.replace('_{}'.format(img_num), ' ({})'.format(img_num))
            img_aaron = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE)
            img_converted = cv2.imread(os.path.join(converted_folder, name, img_converted_name), cv2.IMREAD_GRAYSCALE)

            print_img_stats(img_aaron, img_converted, visualise=True)
    return


def print_img_stats(img1, img2, visualise=False):
    print('num_equal={}; mean1={}; std1={}; mean2={}; std2={}', np.sum(img1 == img2), np.mean(img1), np.std(img1), np.mean(img2), np.std(img2))
    if visualise:
        plt.figure(1)
        plt.clf()
        plt.subplot(131)
        plt.imshow(img1)
        plt.subplot(132)
        plt.imshow(img2)
        plt.subplot(133)
        plt.imshow(img1 - img2)
    return


def check_titan_vs_local_conversion(folder):
    titan_fnames = [x for x in os.listdir(folder) if '.png' in x]
    # local_files = [x for x in os.listdir(os.path.join(folder, 'Uninflamed')) if '.png' in x]
    for titan_fname in titan_fnames:
        titan_path = os.path.join(folder, titan_fname)
        local_folder = titan_fname.split()[0]
        local_path = os.path.join(folder, 'Uninflamed', local_folder, titan_fname)
        titan_img = cv2.imread(titan_path, cv2.IMREAD_GRAYSCALE)
        local_img = cv2.imread(local_path, cv2.IMREAD_GRAYSCALE)
        print_img_stats(titan_img, local_img, visualise=True)
    return


if __name__ == '__main__':
    # convert_stack()
    # check_converted_stack()

    # # compare segmented vs converted
    # check_titan_vs_local_conversion(folder=os.path.join(base_folder, 'Volume_converted'))
    # compare_seg_vs_converted(seg_folder=seg_path, converted_folder=CONVERTED_FOLDER)    # windows conversion seems different

    # show thresholded points give visual illusion of being bigger when overlaid
    # pred_folder = '{}\{}'.format(CONVERTED_FOLDER, '20170710mouse1_Day-7_Left')
    # visualise_preds(folder=pred_folder)

    params = load_params()
    predict_converted_stacks()

    # TODO - implement if needed
    # make_stack_data()
    # train_stack()