import sys, os, glob, cv2, json, subprocess
import numpy as np
import random

from matplotlib import pyplot as plt
from matplotlib import patches
from scipy import ndimage
from data import slice_data
from analyser import combine_img, center_scale_imgs

# make textfile for cones
# filepath, x1, y1, x2, y2, class_name`
# For example:
# /data/imgs/img_001.jpg, 837, 346, 981, 456, cow
# /data/imgs/img_002.jpg, 215, 312, 279, 391, cat

# img folders
train_imgs_folder = './final/train'
valid_imgs_folder = './final/valid'
from sys import platform
if platform == "linux" or platform == "linux2":
    plt.switch_backend('agg')
    # linux
    prefix = os.path.join('/data', 'yue', 'pepple')
    # segmentation_json_folder = '/home/ayl/data/pepple/accell/jsons'
    # img_folder = '/home/ayl/data/pepple/accell/segmentations'
    segmentation_json_folder = '/data/yue/pepple/accell/jsons'
    img_folder = '/data/yue/pepple/accell/segmentations'
    # empty_img_folder = '/home/ayl/data/pepple/accell/empty_segmentations'

    npy_data_folder = '/data/yue/pepple/accell/npy_data'
    # chamber_weights_folder = '/home/yue/pepple/runs/2017-08-09-10-20-24/weights'

    # includes more (unique) images
    chamber_weights_folder = './runs/2017-12-12-17-00-53'
    # test_weights = 'weights-improvement-174--0.83855137.hdf5'
    # test_weights = 'weights-improvement-256--0.85407762.hdf5'
elif platform == "win32":
    # Windows...
    prefix = os.path.join('z:/', 'yue', 'pepple')
    segmentation_json_folder = './accell/jsons'
    img_folder = './accell/segmentations'
    empty_img_folder = './accell/empty_segmentations'

    npy_data_folder = './accell/npy_data'
    # chamber_weights_folder = './runs/runVertical/weights'
    chamber_weights_folder = './runs/runAllUnique'

# test_weights = 'weights-improvement-050--0.95407502.hdf5'
test_weights = 'weights-improvement-174--0.83855137.hdf5'

if not os.path.exists(npy_data_folder):
    os.makedirs(npy_data_folder)

RAW_IMG_ROWS = 1024
RAW_IMG_COLS = 1000
SCALE_FACTOR = 0.5
DOWNSAMPLE_RATIO = 1/SCALE_FACTOR
SCALED_IMG_ROWS = int(RAW_IMG_ROWS * SCALE_FACTOR)
SCALED_IMG_COLS = int(RAW_IMG_COLS * SCALE_FACTOR)
SCALED_IMG_COLS_PADDED = int(RAW_IMG_ROWS * SCALE_FACTOR)
ACCELL_DIAMETER = 5
MULTI_CLASS=False


# get labelled data file names
def get_img_names(folder, ext='.png'):
    img_names_file = os.path.join(folder, 'img_names.txt')
    if os.path.isfile(img_names_file):
        img_names = []
        with open(img_names_file, 'r') as fin:
            for l in fin:
                img_names.append(l.rstrip())
        fin.close()
    else:
        # if saved already then load
        img_names = [x for x in sorted(os.listdir(folder)) if ext in x.lower() and 'mask' not in x.lower()]
        with open(img_names_file, 'w') as fout:
            for img_name in img_names:
                fout.write('{}\n'.format(img_name))
        fout.close()
    return img_names


# load raw images (1024*1000)
def get_raw_imgs(folder, ext='.png'):
    img_names = get_img_names(folder, ext)
    output_npy = os.path.join(npy_data_folder, 'raw_images.npy')

    # out_folder = os.path.join(folder, 'npy_data')
    # output_npy = os.path.join(out_folder, 'raw_images.npy')     # for that folder
    if os.path.isfile(output_npy):
        raw_images = np.load(output_npy)
        return raw_images, img_names

    # otherwise read and save
    num_images = len(img_names)
    # this assumes all images are same size, which might not hold
    raw_images = np.ndarray((num_images, RAW_IMG_ROWS, RAW_IMG_COLS), dtype=np.float32)
    for idx, img_name in enumerate(img_names):
        raw_images[idx, ] = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
    np.save(output_npy, raw_images)

    return raw_images, img_names


# subprocess call imagemagick to scale images
def convert_image(tiff_dir, tiff_name, tiff_out_dir, options='-scale 50%', debug=False):
    # convert -scale 50% "#{f}" "#{fout}
    image_magick_cmd = '{} {}'.format('convert', options)
    tiff_in_path = os.path.join(tiff_dir, tiff_name)
    tiff_out_path = os.path.join(tiff_out_dir, tiff_name.replace('.TIFF', '.png'))

    shell_arguments = [image_magick_cmd, tiff_in_path, tiff_out_path]
    print(shell_arguments)
    # subprocess.call([image_magick_cmd, tiff_in_path, tiff_out_path])
    subprocess.call(' '.join([image_magick_cmd, '"{}"'.format(tiff_in_path), '"{}"'.format(tiff_out_path)]), shell=True)

    if debug:
        img_orig = cv2.imread(tiff_in_path, cv2.IMREAD_GRAYSCALE)
        img_scaled =cv2.imread(tiff_out_path, cv2.IMREAD_GRAYSCALE)
        print(tiff_in_path, img_orig.shape, tiff_out_path, img_scaled.shape)
    return tiff_out_path


# scale images with imagemagick from 1024*1000 to 512*500
def convert_raw_imgs(folder, out_folder=npy_data_folder, ext='.tiff'):
    raw_images, img_names = get_raw_imgs(folder, ext=ext)
    converted_npy = os.path.join(npy_data_folder, 'converted_imgs.npy')
    if os.path.isfile(converted_npy):
        converted_imgs = np.load(converted_npy)
        return raw_images, converted_imgs, img_names

    num_images = raw_images.shape[0]    # NB - this might be different if not all 1200 slices used
    # always same size?
    converted_imgs = np.ndarray((num_images, SCALED_IMG_ROWS, SCALED_IMG_COLS), dtype=np.float32)
    output_dir = os.path.join(folder, 'seg_scaled')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # convert raw images using imagemagick, read and save
    for idx, img_name in enumerate(img_names):
        converted_img_name = convert_image(folder, img_name, output_dir)    # rescale image
        converted_imgs[idx, ] = cv2.imread(converted_img_name, cv2.IMREAD_GRAYSCALE)
    np.save(converted_npy, converted_imgs)

    return raw_images, converted_imgs, img_names


# predict on strips and recombine into whole image
def predict_imgs(model, img_data):
    sliced_img, _ = slice_data(img_data, img_data, save_data=False)
    output = model.predict(sliced_img, batch_size=10)  # inference step
    num_aug = 96

    predicted_shape = output.shape
    num_images = int(predicted_shape[0] / num_aug)
    all_combined_preds = np.ndarray((num_images, SCALED_IMG_ROWS, SCALED_IMG_COLS_PADDED), dtype=np.float32)
    for idx in range(0, predicted_shape[0], num_aug):
        cur_strip_preds = output[idx:idx + num_aug, ]
        cur_combined_pred = combine_img(cur_strip_preds, real_mean=True)  # this pads bottom
        cur_count = int(idx / num_aug)
        all_combined_preds[cur_count, ] = cur_combined_pred
    return all_combined_preds


# load raw_imgs, converted_imgs (scaled), (corresponding) img_names, predict for each converted_img
def read_coords(coord_file):
    fin = open(coord_file).read()
    json_data = json.loads(fin)
    return json_data


def get_img_predictions(folder=img_folder, out_folder=npy_data_folder):
    raw_images, converted_imgs, img_names = convert_raw_imgs(folder, out_folder, ext='.png')
    # return raw_images, converted_imgs, img_names, []    # hack to avoid doing prediction here

    pred_npy = os.path.join(out_folder, 'img_preds.npy')
    if os.path.isfile(pred_npy):
        img_preds = np.load(pred_npy)
        return raw_images, converted_imgs, img_names, img_preds

    from train import get_unet
    model = get_unet()
    weight_file = os.path.join(chamber_weights_folder, 'weights', test_weights)  # use 50th epoch for sanity
    print('model weight file', weight_file)
    model.load_weights(weight_file)

    num_images, img_rows, img_cols = converted_imgs.shape
    zero_pad = np.zeros((num_images, img_rows, img_rows - img_cols), dtype=np.float32)
    converted_imgs = np.concatenate((converted_imgs, zero_pad), axis=2)
    img_preds = predict_imgs(model, converted_imgs.reshape((num_images, img_rows, img_rows, 1)))
    np.save(pred_npy, img_preds)
    # save img_names for further reading
    return raw_images, converted_imgs, img_names, img_preds


# organize coords into more manageable data structure
def get_coords(coord_file):
    coord_data = read_coords(coord_file)
    coords = []
    for coord in coord_data:
        xs = coord['mousex']
        ys = coord['mousey']
        for idx, x in enumerate(xs):
            coords.append((x, ys[idx]))
    return coords


# box boundaries given starting coord, respecting img boundaries (img_shape)
def make_box_coords(coord, img_shape, box_size=ACCELL_DIAMETER):
    cur_x, cur_y = coord
    img_height, img_width = img_shape   # this is y,x

    lower_x = max(cur_x - (box_size // 2), 0)
    upper_x = min(cur_x + (box_size // 2)+1, img_width)

    lower_y = max(cur_y - (box_size // 2), 0)
    upper_y = min(cur_y + (box_size // 2)+1, img_height)
    return lower_x, lower_y, upper_x, upper_y


# from predicted images: 1. rescale, 2. remove segmented cells 3. by resampling segmented ac chamber, 4. store stats
def get_scrubbed_imgs(folder, accell_json_folder=segmentation_json_folder, do_avg=False, visualise=False):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)
        if visualise:
            k = np.random.randint(0, len(raw_images), 1)[0]  # visualise random raw vs averaged
            plt.figure(1)
            plt.clf()
            plt.subplot(131)
            plt.imshow(raw_images[k,])
            plt.title('averaged {}'.format(img_names[k]))
            plt.subplot(132)
            plt.imshow(raw_images_old[k,])
            plt.title('raw {}'.format(img_names[k]))
            plt.subplot(133)
            temp = raw_images[k,] - raw_images_old[k,]
            print('diff_q={}'.format(np.percentile(temp, q=[0, 5, 10, 50, 90, 95, 100])))
            plt.imshow(temp)
            plt.title('diff {}'.format(img_names[k]))

    cleaned_npy = os.path.join(npy_data_folder, 'cleaned_imgs{}.npy'.format('' if do_avg else '_1scan'))
    chamber_stats_npy = os.path.join(npy_data_folder, 'chamber_stats{}.npy'.format('' if do_avg else '_1scan'))
    if os.path.isfile(cleaned_npy) and os.path.isfile(chamber_stats_npy):
        cleaned_imgs = np.load(cleaned_npy)
        chamber_stats = np.load(chamber_stats_npy)  # can get img stats directly - so no need to compute here
        return raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats

    num_images, img_rows, img_cols = converted_imgs.shape   # 512*500
    if not do_avg:  # raw images - not the avg ones
        raw_images_old = np.copy(raw_images)    # since raw_images are going to be changed

    chamber_stats = np.ndarray((num_images, 3), dtype=np.float32)
    patch_size = ACCELL_DIAMETER+2  # slightly bigger to be more conservative with scrubbing
    for idx, img_name in enumerate(img_names):
        print('scrubbing {}'.format(img_name))
        # get corresponding ac coords for img
        # coord_file = os.path.join(accell_json_folder, img_name.replace('.png', '.json'))
        if do_avg:
            coord_file = os.path.join('{}_recentered'.format(accell_json_folder), img_name.replace('.png', '.json'))
        else:
            coord_file = os.path.join('{}_recentered_1scan'.format(accell_json_folder), img_name.replace('.png', '.json'))
        coords = get_coords(coord_file)
        cur_img = raw_images[idx, ]     # want cleaned_imgs on original scale (1024*1000)
        temp_img = np.copy(cur_img)
        img_shape = cur_img.shape
        img_pred = img_preds[idx, ]     # predictions are 512*512
        scaled_img = converted_imgs[idx, ]
        for coord in coords:    # scrub each coord
            # get a patch in chamber to replace segmented accell
            rand_patch, patch_coords, scaled_coords, chamber_mean, chamber_std, chamber_shape = \
                img_sampler(cur_img, scaled_img, img_pred, patch_size=patch_size, centralize_factor=1, all_in=True)
            x_lower, y_lower, x_upper, y_upper = make_box_coords(coord, img_shape=img_shape, box_size=patch_size)
            cur_img[y_lower:y_upper, x_lower:x_upper] = rand_patch
            print('rand_patch matches origin', np.sum(rand_patch == cur_img[patch_coords[1]:patch_coords[3], patch_coords[0]:patch_coords[2]]))
        chamber_stats[idx, ] = [chamber_mean, chamber_std, chamber_shape[0]]    # after all cells removed

        # check scrubbing removes segmented cell
        if visualise:
            print('#pixels same', np.sum(temp_img == cur_img), np.prod(img_shape), len(coords), np.prod(img_shape)-len(coords)*patch_size**2)
            plt.figure(1)
            plt.clf()
            plt.imshow(rand_patch)
            print('rand_patch stats', get_patch_stats(rand_patch))
            plt.figure(2)
            plt.clf()
            plt.imshow(cur_img)
            plt.scatter(x=[patch_coords[0], patch_coords[0], patch_coords[2], patch_coords[2]],
                        y=[patch_coords[1], patch_coords[3], patch_coords[1], patch_coords[3]], c='r', s=2)
            plt.title('scrubbed image with rand_patch')
            plt.figure(3)
            plt.clf()
            plt.imshow(temp_img)
            plt.title('raw image with segmented cells')
            np_coords = np.asarray(coords)
            plt.scatter(x=np_coords[:, 0], y=np_coords[:, 1], c='r', s=2)

    np.save(cleaned_npy, raw_images)
    np.save(chamber_stats_npy, chamber_stats)
    return raw_images_old, converted_imgs, img_names, img_preds, raw_images, chamber_stats


def get_patch_stats(patch):
    return np.mean(patch), np.std(patch), patch.shape


def find_chamber_center(img_pred, pred_threshold=.9):
    # pred_threshold = 0.9  # very conservative threshold
    chamber_limits = np.argwhere(img_pred > pred_threshold)  # n*2 where either 1st/2nd coord below pred_threshold

    # mean_y, mean_x = np.mean(chamber_limits, axis=0)
    if len(chamber_limits)>0:
        # better way to find center
        mean_x = np.mean(chamber_limits[:, 1])  # careful about meaning of coordinates: 1st coord is y-axis
        # mean_y = np.percentile(chamber_limits[:, 0], q=25)
        mean_y = np.mean(chamber_limits[chamber_limits[:, 1] == int(mean_x), 0])  # mid-y for central x
    else:
        mean_x = 0
        mean_y = 0
    return chamber_limits, mean_x, mean_y


# just checks corners are in segmented limits
def is_corners_valid(box_coords, chamber_limits):
    x1, y1, x2, y2 = box_coords
    corners = [(y1, x1), (y2, x1), (y1, x2), (y2, x2)]  # careful of order in chamber_limits
    all_in = True
    for corner in corners:
        corner_idx = np.where((chamber_limits == corner).all(axis=1))[0]
        all_in = all_in and len(corner_idx) > 0
    return all_in


# check all patch coords are in segmented limits
def is_valid_patch(box_coords, chamber_limits):
    x1, y1, x2, y2 = box_coords
    all_in = True
    for x in range(x1, x2+1, 1):
        for y in range(y1, y2+1, 1):
            corner = (y, x)     # careful about axis orientation
            corner_idx = np.where((chamber_limits == corner).all(axis=1))[0]
            all_in = all_in and len(corner_idx) > 0
            if not all_in:  # early circuit breaker
                return all_in
    return all_in


# sample random patch from img
def in_ac_chamber(x, y, chamber_limits, patch_size=32):
    # x, y represent center
    x1 = int(x - patch_size /DOWNSAMPLE_RATIO / 2)
    x2 = int(x + patch_size /DOWNSAMPLE_RATIO / 2)
    y1 = int(y - patch_size /DOWNSAMPLE_RATIO / 2)
    y2 = int(y + patch_size /DOWNSAMPLE_RATIO / 2)
    # NB coords swapped and nearest point in chamber_coords (512*500)
    if [y1, x1] in chamber_limits.tolist() and [y2, x2] in chamber_limits.tolist():
        return True
    else:
        return False


def get_rand_patch_helper(chamber_limits, mean_x, mean_y, patch_size=ACCELL_DIAMETER, centralize_factor=1):
    # random point in thresholded prediction (scaled/downsampled) area
    box_idx = random.randint(0, chamber_limits.shape[0] - 1)  # random point inside mask 512*512
    cur_coord = chamber_limits[box_idx, ]
    cur_y, cur_x = cur_coord

    # move towards image center - if centralize_factor > 0
    cur_x = cur_x + centralize_factor * patch_size / 2. * np.sign(mean_x - cur_x)
    cur_y = cur_y + centralize_factor * patch_size / 2. * np.sign(mean_y - cur_y)

    if not in_ac_chamber(x=cur_x, y=cur_y, chamber_limits=chamber_limits, patch_size=patch_size):  # recursive
        cur_x, cur_y = get_rand_patch_helper(chamber_limits, mean_x, mean_y, patch_size, centralize_factor)

    return cur_x, cur_y


def get_rand_patch(cur_img, chamber_limits, mean_x, mean_y, patch_size=ACCELL_DIAMETER, centralize_factor=1):
    cur_x, cur_y = get_rand_patch_helper(chamber_limits, mean_x, mean_y, patch_size, centralize_factor)

    x1 = int(cur_x * DOWNSAMPLE_RATIO - patch_size / 2.)  # rescale
    x2 = int(cur_x * DOWNSAMPLE_RATIO + patch_size / 2.)
    y1 = int(cur_y * DOWNSAMPLE_RATIO - patch_size / 2.)
    y2 = int(cur_y * DOWNSAMPLE_RATIO + patch_size / 2.)

    patch_coords = (x1, y1, x2, y2)
    scaled_coords = (int(x1/DOWNSAMPLE_RATIO), int(y1/DOWNSAMPLE_RATIO), int(x2/DOWNSAMPLE_RATIO), int(y2/DOWNSAMPLE_RATIO))

    cur_patch = cur_img[y1:y2, x1:x2]
    return cur_patch, scaled_coords, patch_coords


# FIXME - can be more efficient by passing chamber_limits, mean_x, mean_y
# recursively sample until patch meets conditions
def img_sampler(cur_img, scaled_img, img_pred, patch_size=ACCELL_DIAMETER, centralize_factor=1, all_in=True):
    chamber_limits, mean_x, mean_y = find_chamber_center(img_pred)
    chamber_intensities = scaled_img[chamber_limits[:, 0], chamber_limits[:, 1]]
    cur_patch, scaled_coords, patch_coords \
        = get_rand_patch(cur_img, chamber_limits, mean_x, mean_y, patch_size, centralize_factor)
    # np.where((chamber_limits == (int((scaled_coords[1]+scaled_coords[3])/2), int((scaled_coords[0]+scaled_coords[2])/2))).all(axis=1))

    # check all patch coordinates in thresholded limits
    patch_mean, patch_std, patch_shape = get_patch_stats(cur_patch)
    chamber_mean, chamber_std, chamber_shape = get_patch_stats(chamber_intensities)     # approximate for scaled image

    # intensity_condition = patch_mean > min(chamber_mean + 2*chamber_std/patch_size, 40)    # NB 5 > mean_std=pixel_std/sqrt(32*32)
    intensity_condition = patch_mean > min(chamber_mean + 5, 40)
    if intensity_condition:     # warn about intensities
        print('patch_mean={} is iffy. chamber_mean={}; patch_std={}; chamber_std={}, chamber_shape={}; patch_size={}'
              .format(patch_mean, chamber_mean, patch_std, chamber_std, chamber_shape, patch_size))
        # visualize_patch_on_original(cur_img, scaled_img, scaled_coords, cur_patch, patch_coords)
        # visualize_img_prediction(cur_img, scaled_img, img_pred, chamber_limits)

    if is_valid_patch(scaled_coords, chamber_limits) and not intensity_condition:
        return cur_patch, patch_coords, scaled_coords, chamber_mean, chamber_std, chamber_shape
    else:
        return img_sampler(cur_img, scaled_img, img_pred, patch_size, centralize_factor, all_in)


def visualize_img_prediction(img, scaled_img, scaled_pred, chamber_limits=None):
    if chamber_limits is not None:
        chamber_limits, mean_x, mean_y = find_chamber_center(scaled_pred)
    plt.figure(1)
    plt.clf()
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(scaled_pred)
    plt.subplot(133)
    plt.imshow(scaled_img)
    plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='yellow', s=1)

    # k = 100;
    # plt.figure(100);
    # plt.clf();
    # plt.subplot(131);
    # plt.imshow(raw_images[k,]);
    # plt.subplot(132);
    # plt.imshow(converted_imgs[k,]);
    # plt.subplot(133);
    # plt.imshow(img_preds[k,])
    return


def visualize_patch_on_original(img, scaled_img, scaled_coords, patch, patch_coords):
    x_lower_scaled, y_lower_scaled, x_upper_scaled, y_upper_scaled = scaled_coords
    x_center_scaled, y_center_scaled = int((x_lower_scaled+x_upper_scaled)/2), int((y_lower_scaled+y_upper_scaled)/2)
    x_lower, y_lower, x_upper, y_upper = patch_coords
    x_center, y_center = int((x_lower + x_upper) / 2), int((y_lower + y_upper) / 2)
    plt.figure(1)
    plt.clf()
    plt.imshow(img)
    plt.scatter(x=x_center, y=y_center, c='red', s=5)  # coord center
    plt.axes().add_patch(patches.Rectangle((x_lower, y_lower), x_upper-x_lower, y_upper-y_lower, fill=False, color='green'))  # show box
    plt.title('patch on original img')

    plt.figure(2)
    plt.clf()
    plt.imshow(scaled_img)
    plt.scatter(x=x_center_scaled, y=y_center_scaled, c='red', s=5)  # coord center
    plt.axes().add_patch(patches.Rectangle((x_lower_scaled, y_lower_scaled), x_upper_scaled - x_lower_scaled,
                                           y_upper_scaled - y_lower_scaled, fill=False, color='green'))  # show box
    plt.title('patch on scaled img')
    return


def parse_img_base_name(pname):
        if platform == "linux" or platform == "linux2":
            p_toks = pname.split('/')
        elif platform == "win32":
            p_toks = pname.split('\\')
        sample_name = p_toks[-1].replace('.png', '')
        return sample_name


# wrapper for getting imgs and averaged versions
def get_img_preds_wrapper(img_folder, do_avg=False, visualise=False):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(img_folder)
    # raw_images, img_names = get_raw_imgs(img_folder, ext='.png')
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)
        if visualise:
            k = np.random.randint(0, len(raw_images), 1)[0]  # visualise random raw vs averaged
            plt.figure(1)
            plt.clf()
            plt.subplot(131)
            plt.imshow(raw_images[k,])
            plt.title('averaged {}'.format(img_names[k]))
            plt.subplot(132)
            plt.imshow(raw_images_old[k,])
            plt.title('raw {}'.format(img_names[k]))
            plt.subplot(133)
            temp = raw_images[k,] - raw_images_old[k,]
            print('diff_q={}'.format(np.percentile(temp, q=[0, 5, 10, 50, 90, 95, 100])))
            plt.imshow(temp)
            plt.title('diff {}'.format(img_names[k]))
    # return raw_images, img_names
    return raw_images, converted_imgs, img_names, img_preds


# grab and store all accell patches and chamber info
def get_all_ac_cells(seg_folder, img_folder, do_avg=False, visualise=False):
    cell_labels = []
    cell_size = ACCELL_DIAMETER
    if do_avg:  # averaged scans
        recentered_folder = '{}_recentered'.format(seg_folder)
    else:
        recentered_folder = '{}_recentered_1scan'.format(seg_folder)

    # raw_images, img_names = get_img_preds_wrapper(img_folder, do_avg=do_avg, visualise=visualise)
    # need preds to avoid going onto chamber border
    raw_images, converted_imgs, img_names, img_preds = get_img_preds_wrapper(img_folder, do_avg=do_avg, visualise=visualise)
    if os.path.isdir(recentered_folder) and len([x for x in os.listdir(recentered_folder) if '.json' in x])>=len(raw_images):
        1   # already recentered - continue
    else:
        recenter_cells_for_images(raw_images, img_names, img_preds, seg_folder, do_avg=do_avg)    # raw_images is avg or raw png

    # keep other framework for speed and debugging purposes
    raw_images2, converted_imgs2, img_names2, img_preds2, cleaned_imgs, chamber_stats = get_scrubbed_imgs(img_folder, do_avg=do_avg)

    if do_avg:
        cell_npy = '{}/cell_{}.npy'.format(npy_data_folder, cell_size)
        cell_stats_npy = '{}/cell_stats_{}.npy'.format(npy_data_folder, cell_size)
    else:
        cell_npy = '{}/cell_{}_1scan.npy'.format(npy_data_folder, cell_size)
        cell_stats_npy = '{}/cell_stats_{}_1scan.npy'.format(npy_data_folder, cell_size)

    if os.path.isfile(cell_npy) and os.path.isfile(cell_stats_npy):
        all_cell_data = np.load(cell_npy)
        all_cell_stats = np.load(cell_stats_npy)

        with open(os.path.join(npy_data_folder, 'cell_labels{}.txt'.format('' if do_avg else '_1scan')), 'r') as fin:
            for l in fin.readlines():
                cell_labels.append(l.rstrip())
        fin.close()
    else:
        all_cell_data = np.ndarray((0, cell_size, cell_size), dtype=np.uint8)
        all_cell_stats = []

        # avg_intensity = np.mean(raw_images)   # adjust for average intensity
        for idx, img_name in enumerate(img_names):  # get accells for segmented image
            sample_name = parse_img_base_name(img_name)
            json_path = '{}/{}.json'.format(recentered_folder, sample_name)  # get fixed cells
            coords = get_coords(json_path)
            cur_image = raw_images[idx, ]   # since accell segmentation on 1024*1000 png
            img_intensity_mean, img_intensity_std, img_shape = get_patch_stats(cur_image)   # img_stats - probably less relevant than chamber_stats

            img_cell_data = np.zeros((len(coords), cell_size, cell_size), dtype=np.uint8)
            for jdx, coord in enumerate(coords):
                cell_labels.append(img_name)    # add label for each cell in each image

                cur_x, cur_y = coord
                # already recentered since grabbing from recentered_folder
                # cur_cell, cell_coords, old_cell = recenter_cell(cur_image, cur_x, cur_y, img_shape)     # update cell jsons after re-centering
                cur_cell, cell_box_coords = get_cell(cur_image, cur_x, cur_y, img_shape, cell_size=cell_size)
                img_cell_data[jdx,] = cur_cell
                cell_intensity_mean, cell_intensity_std, cell_shape = get_patch_stats(cur_cell)
                cur_cell_stats = (img_intensity_mean, img_intensity_std, cell_intensity_mean, cell_intensity_std,
                                  chamber_stats[idx, 0], chamber_stats[idx, 1])
                all_cell_stats.append(cur_cell_stats)
            all_cell_data = np.append(all_cell_data, img_cell_data, axis=0)

        np.save(cell_npy, all_cell_data)
        all_cell_stats = np.asarray(all_cell_stats, dtype=np.float32)
        np.save(cell_stats_npy, all_cell_stats)

        with open(os.path.join(npy_data_folder, 'cell_labels{}.txt'.format('' if do_avg else '_1scan')), 'w') as fout:
            for cell_label in cell_labels:
                fout.write('{}\n'.format(cell_label))
        fout.close()

    # some percentile stats for cell data
    print(np.percentile(all_cell_stats, [5, 10, 25, 50, 75, 90, 95], axis=0))
    print(np.percentile(all_cell_stats[:, 2], [5, 10, 25, 50, 75, 90, 95]))
    return all_cell_data, all_cell_stats, cell_labels


# seg_folder with human accell segmentations
def recenter_cells_for_images(raw_images, img_names, img_preds, seg_folder, cell_size=ACCELL_DIAMETER, do_avg=True):
    if do_avg:
        recentered_folder = '{}_recentered_{}by{}'.format(seg_folder, cell_size, cell_size)
    else:
        recentered_folder = '{}_recentered_1scan_{}by{}'.format(seg_folder, cell_size, cell_size)

    if not os.path.isdir(recentered_folder):
        os.makedirs(recentered_folder)

    all_cell_data = np.ndarray((0, cell_size, cell_size), dtype=np.uint8)
    cell_labels = []
    for idx, img_name in enumerate(img_names):  # get accells for segmented image
        cell_labels.append(img_name)

        recentered_coords = []     # for output
        sample_name = parse_img_base_name(img_name)
        json_path = '{}/{}.json'.format(seg_folder, sample_name)
        coords = get_coords(json_path)
        cur_image = raw_images[idx, ]  # since accell segmentation on 1024*1000 png
        cur_pred = img_preds[idx, ]     # NB - on smaller scale 512*512
        img_intensity_mean, img_intensity_std, img_shape = get_patch_stats(cur_image)  # img_stats - probably less relevant than chamber_stats

        img_cell_data = np.zeros((len(coords), cell_size, cell_size), dtype=np.uint8)
        for jdx, coord in enumerate(coords):
            cur_x, cur_y = coord
            cur_cell, cell_coords, old_cell = recenter_cell(cur_image, cur_x, cur_y, img_shape, img_pred=cur_pred, img_name=sample_name, cell_size=cell_size)
            cell_intensity_mean, cell_intensity_std, cell_shape = get_patch_stats(cur_cell)

            x1, y1, x2, y2 = cell_coords
            recentered_coords.append({"mousex":[int((x1+x2)/2.0)], "mousey":[int((y1+y2)/2.0)], "mousetime":[]})

        all_cell_data = np.append(all_cell_data, img_cell_data, axis=0)
        # write to recentered folder
        # '[{"mousex":[182],"mousey":[445],"mousetime":[]}]'
        with open(os.path.join(recentered_folder, img_name.replace('png', 'json')), 'w') as fout:
            json.dump(recentered_coords, fout)
        fout.close()
    1
    return all_cell_data, cell_labels


def get_max_coord_from_patch(img):
    # cur_coord = img.argmax(axis=0)
    i, j = np.unravel_index(img.argmax(), img.shape)
    return i, j


# NB - this can break near chamber edges
def get_best_local_cell(img, cur_x, cur_y, num_pixels=2, img_pred=None, img_name=None, cell_size=ACCELL_DIAMETER, visualise=False):   # num_pixels pixels to consider in each direction
    image_shape = img.shape
    avg_intensities = np.zeros((num_pixels*2+1, num_pixels*2+1), dtype=np.float32)
    search_range = range(-num_pixels, num_pixels+1, 1)  # num_pixels pixels to consider in each direction
    for idx, dx in enumerate(search_range):
        for jdy, dy in enumerate(search_range):
            new_x = cur_x + dx
            new_y = cur_y + dy
            new_cell_coords = make_box_coords((new_x, new_y), image_shape, box_size=cell_size)
            new_cell = img[new_cell_coords[1]:new_cell_coords[3], new_cell_coords[0]:new_cell_coords[2]]
            avg_intensities[idx, jdy] = np.mean(new_cell)

    i, j = np.unravel_index(avg_intensities.argmax(), avg_intensities.shape)
    best_x = cur_x + search_range[i]
    best_y = cur_y + search_range[j]
    best_cell_coords = make_box_coords((best_x, best_y), image_shape, box_size=cell_size)
    best_cell = img[best_cell_coords[1]:best_cell_coords[3], best_cell_coords[0]:best_cell_coords[2]]

    if img_pred is not None:
        # using lower threshold makes boundaries bigger, but more conservative in terms where cells can be moved
        chamber_limits, mean_x, mean_y = find_chamber_center(img_pred, pred_threshold=.2)
        if visualise:
            plt.figure(100)
            plt.clf()
            plt.imshow(img)
            plt.scatter(x=chamber_limits[:,1]*DOWNSAMPLE_RATIO, y=chamber_limits[:,0]*DOWNSAMPLE_RATIO, c='y', s=1)

        scaled_x = int(best_x/DOWNSAMPLE_RATIO)   # scale appropriately
        scaled_y = int(best_y/DOWNSAMPLE_RATIO)
        if not in_ac_chamber(scaled_x, scaled_y, chamber_limits, patch_size=cell_size):  # reject if overlapping with chamber border
            print('on border', img_name, best_x, best_y, scaled_x, scaled_y)
            best_x = cur_x  # revert to human seg
            best_y = cur_y
            best_cell_coords = make_box_coords((best_x, best_y), image_shape, box_size=cell_size)
            best_cell = img[best_cell_coords[1]:best_cell_coords[3], best_cell_coords[0]:best_cell_coords[2]]

    if visualise:
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img.reshape(RAW_IMG_ROWS, RAW_IMG_COLS))
        ax1.scatter(x=cur_x, y=cur_y, c='white', s=2)  # not flipped
        (init_x1, init_y1, init_x2, init_y2) = make_box_coords((cur_x, cur_y), image_shape, box_size=cell_size)
        ax1.add_patch(patches.Rectangle((init_x1, init_y1), cell_size, cell_size, fill=False, color='white'))
        ax1.add_patch(patches.Rectangle((best_cell_coords[0], best_cell_coords[1]), cell_size, cell_size, fill=False, color='red'))

    return best_cell, best_cell_coords, best_x, best_y


def get_cell(img, cur_x, cur_y, image_shape, cell_size=ACCELL_DIAMETER):
    init_cell_coords = make_box_coords((cur_x, cur_y), image_shape, box_size=cell_size)
    init_cell = img[init_cell_coords[1]:init_cell_coords[3], init_cell_coords[0]:init_cell_coords[2]]  # NB. meaning of coords
    return init_cell, init_cell_coords


def recenter_cell(img, cur_x, cur_y, image_shape, img_pred=None, cell_size=ACCELL_DIAMETER, img_name=None, visualise=False):
    init_cell, init_cell_coords = get_cell(img, cur_x, cur_y, image_shape, cell_size=cell_size)
    # cur_cell = img[cur_cell_coords[0]:cur_cell_coords[2], cur_cell_coords[1]:cur_cell_coords[3], 0]

    # get maximal intensity location and re-center
    old_search = False
    if old_search:
        cell_y, cell_x = get_max_coord_from_patch(init_cell)
        new_y = cur_y + cell_y - cell_size//2
        new_x = cur_x + cell_x - cell_size//2

        new_cell_coords = make_box_coords((new_x, new_y), image_shape, box_size=cell_size)
        new_cell = img[new_cell_coords[1]:new_cell_coords[3], new_cell_coords[0]:new_cell_coords[2], 0]  # NB. meaning of coords
        if np.mean(new_cell) < np.mean(init_cell):  # want best intensity patches
            new_cell = init_cell
            new_cell_coords = init_cell_coords
    else:
        new_cell, new_cell_coords, new_x, new_y = get_best_local_cell(img, cur_x, cur_y, num_pixels=2, img_pred=img_pred, img_name=img_name, cell_size=cell_size)

    if visualise:
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img.reshape(RAW_IMG_ROWS, RAW_IMG_COLS))
        ax1.scatter(x=cur_x, y=cur_y, c='white', s=2)  # not flipped
        (init_x1, init_y1, init_x2, init_y2) = init_cell_coords
        ax1.add_patch(patches.Rectangle((init_x1, init_y1), cell_size, cell_size, fill=False, color='white'))  # x and y axis make sense here

        ax1.scatter(x=new_x, y=new_y, c='red', s=2)  # not flipped
        (real_x1, real_y1, real_x2, real_y2) = new_cell_coords
        ax1.add_patch(patches.Rectangle((real_x1, real_y1), cell_size, cell_size, fill=False, color='red'))
    return np.copy(new_cell), new_cell_coords, np.copy(init_cell)


# create training data of ac cells on background ac chamber patches
def create_accell_data(patch_rows=32, patch_cols=32, num_samples=10000, visualise=False):
    # cell data with stats of img and chamber where they are from
    all_cell_data, all_cell_stats, cell_labels = get_all_ac_cells(segmentation_json_folder, img_folder, visualise=visualise)
    # get imgs, predictions and cleaned imgs without accell data
    raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats = get_scrubbed_imgs(img_folder)

    # only take cleaned_imgs with largist chambers for sampling patch
    raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats \
        = get_larger_chambers(img_folder, accell_json_folder=segmentation_json_folder)
    num_imgs = raw_images.shape[0]

    out_folder_name = 'ac_training3_{}_{}'.format(patch_rows, patch_cols)
    if platform == "linux" or platform == "linux2":
        base_folder = '/home/yue/pepple/accell/{}'.format(out_folder_name)
    else:
        base_folder = './accell/{}'.format(out_folder_name)

    train_folder = '{}/train'.format(base_folder)
    valid_folder = '{}/valid'.format(base_folder)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        os.makedirs(train_folder)
        os.makedirs(valid_folder)

    # about 2500 cells - split appropriately;
    # FIXME - could system memorize cells (probably)
    train_valid_split = .8
    num_coords_train = int(len(all_cell_data) * train_valid_split)
    cell_train = all_cell_data[:num_coords_train, ]
    cell_train_stats = all_cell_stats[:num_coords_train, ]
    create_accell_imgs(cleaned_imgs, converted_imgs, img_preds, chamber_stats, cell_train, cell_train_stats,
                       base_folder, num_samples, patch_rows, patch_cols, is_train=True, visualise=visualise)
    # validation data
    cell_valid = all_cell_data[num_coords_train:, ]
    cell_valid_stats = all_cell_stats[num_coords_train:, ]
    num_test = int(num_samples*.1)
    create_accell_imgs(cleaned_imgs, converted_imgs, img_preds, chamber_stats, cell_valid, cell_valid_stats,
                       base_folder, num_test, patch_rows, patch_cols, is_train=False, visualise=visualise)
    return


def create_accell_imgs(cleaned_imgs, converted_imgs, img_preds, chamber_stats, cells_data, cells_stats, output_folder,
                       num_samples=10000, patch_rows=32, patch_cols=32, is_train=True, do_blurred=False, visualise=False):
    num_imgs = cleaned_imgs.shape[0]

    for idx in range(num_samples):
        # pick random img
        sample_idx = random.randint(0, num_imgs - 1)
        cur_img = cleaned_imgs[sample_idx, ]
        scaled_img = converted_imgs[sample_idx, ]
        img_pred = img_preds[sample_idx, ]
        rand_patch, patch_coords, scaled_coords, chamber_mean, chamber_std, chamber_shape \
            = img_sampler(cur_img, scaled_img, img_pred, patch_size=patch_rows, centralize_factor=1, all_in=True)
        aug_patch, coords, mid_coords, cell_types = create_accell_img(rand_patch, chamber_stats[sample_idx, ], cells_data, cells_stats, do_blurred=do_blurred)

        if is_train:
            img_path = os.path.join(output_folder, 'train', 'training_{}.png'.format(idx))
        else:
            img_path = os.path.join(output_folder, 'valid', 'test_{}.png'.format(idx))
        cv2.imwrite(img_path, aug_patch)

        if visualise:
            plt.figure(1)
            plt.clf()
            plt.imshow(rand_patch)
            plt.title('rand_patch')
            plt.figure(2)
            plt.clf()
            plt.imshow(cur_img)
            # plt.scatter(x=[patch_coords[0], patch_coords[0], patch_coords[2], patch_coords[0]], y=[patch_coords[1], patch_coords[3], patch_coords[1], patch_coords[3]], c='red', s=2)
            plt.axes().add_patch(patches.Rectangle((patch_coords[0], patch_coords[1]), patch_rows, patch_rows, fill=False, color='red'))  # show box
            plt.title('rand_patch origin')
            print('rand_patch matches origin', np.sum(rand_patch == cur_img[patch_coords[1]:patch_coords[3], patch_coords[0]:patch_coords[2]]))

            plt.figure(3)
            plt.clf()
            plt.imshow(aug_patch)
            plt.scatter(x=mid_coords[:, 0], y=mid_coords[:, 1], c='red', s=1)
            plt.title('patch with cells centers')
            print('rand vs aug patch', np.sum(rand_patch==aug_patch), patch_rows**2)
            print('augmented patch stats', get_patch_stats(aug_patch))

        with open(os.path.join(output_folder, '{}_coords.txt'.format('training' if is_train else 'valid')), 'a') as fout:
            for jdx, coord in enumerate(coords):
                # vals = [img_path, str(coord[0]), str(coord[1]), str(coord[2]), str(coord[3]), 'cell']
                vals = [img_path, str(coord[0]), str(coord[1]), str(coord[2]), str(coord[3]), cell_types[jdx]]
                # print(','.join(vals))
                fout.write('{}\n'.format(','.join(vals)))
    return


# plonk cells onto img/patch
def create_accell_img(raw_img, img_chamber_stats, cells_data, cells_stats, do_blurred=False, visualise=False):
    num_cells, _, _ = cells_data.shape
    num_rows, num_cols = raw_img.shape
    img = np.copy(raw_img)

    # add cells
    cell_size = ACCELL_DIAMETER
    max_cells = num_rows*num_cols / (cell_size * cell_size)
    cell_upper = int(max_cells * .2*.5)     # up to 32*32/25*.3*.5 = 12*.5 = 6 cells per image
    # cell_lower = int(max_cells * .1*.5)    # at least 1 cell in image for useful training
    cell_lower = 2
    print('img_shape={}; max_cells={}; cell_upper={}; cell_lower={}'.format(img.shape, max_cells, cell_upper, cell_lower))
    num_samples = random.randint(cell_lower, cell_upper)    # number of cells to add

    mean_x = num_cols/2.
    mean_y = num_rows/2.
    mid_coords = np.zeros((0, 2), dtype=np.uint8)  # pre-allocate
    coords = np.zeros((0, 4), dtype=np.uint8)  # pre-allocate
    cell_types = []
    for idx in range(num_samples):
        rand_ind = random.choice(range(0, num_cells))
        rand_cell = cells_data[rand_ind, ]
        cell_stats = cells_stats[rand_ind, ]

        # random coord in img patch to plonk cell
        new_y, new_x = random.randint(cell_size//2, num_rows-cell_size//2 - 1), \
                       random.randint(cell_size//2, num_cols-cell_size//2 - 1)

        # avoid overlapping
        mid_coord = (new_x, new_y)
        overlapping_new = is_overlapping(mid_coords, mid_coord)
        if not overlapping_new:
            x_lower, y_lower, x_upper, y_upper = make_box_coords((new_x, new_y), (num_rows, num_cols), ACCELL_DIAMETER)
            new_coord = x_lower, y_lower, x_upper, y_upper
            if img[y_lower:y_upper, x_lower:x_upper].shape != (cell_size, cell_size):   # sanity check
                print('new_x={}; new_y={}; new_coord={}'.format(new_x, new_y, new_coord))
            # img, cell_type = set_cell(img, rand_cell, (x_lower, y_lower, x_upper, y_upper), img_chamber_stats, cell_stats,
            #                augment=True, blur=False)
            img, cell_type = set_cell(img, rand_cell, (x_lower, y_lower, x_upper, y_upper), img_chamber_stats,
                                      cell_stats, augment=True, blur=do_blurred)
            mid_coords = np.append(mid_coords, np.reshape(np.asarray(mid_coord), (1, 2)), axis=0)
            coords = np.append(coords, np.reshape(np.asarray(new_coord), (1, 4)), axis=0)
            cell_types.append(cell_type)

            if visualise:
                plt.figure(1)
                plt.imshow(raw_img)
                plt.title('original patch')
                plt.scatter(x=new_x, y=new_y, c='red', s=2)  # coord center
                plt.axes().add_patch(patches.Rectangle((x_lower, y_lower), cell_size, cell_size, fill=False, color='green'))   # show box

                plt.figure(3)
                plt.imshow(img)
                plt.title('cell added patch')
                plt.scatter(x=new_x, y=new_y, c='red', s=2)  # coord center
                plt.axes().add_patch(patches.Rectangle((x_lower, y_lower), cell_size, cell_size, fill=False, color='green'))   # show box

    return img, coords, mid_coords, cell_types


def is_overlapping(coords, new_coord, cell_size=ACCELL_DIAMETER):
    if len(coords):
        dist = np.sum((coords-new_coord)**2, axis=1)
        return np.any(dist<2*(cell_size**2))
    else:
        return False


# this breaks down when ac chamber is in patch
def calc_cell_class(cell, img, img_pred=None):
    cell_type = 'cell_lite'
    cell_mean = np.mean(cell)

    if img_pred is None:
        img_mean = np.mean(img)
    else:
        pred_threshold = 0.85   # more aggressive threshold
        chamber_limits = np.nonzero(img_pred>pred_threshold)
        chamber_limits_rs = tuple(np.array(np.array(chamber_limits) * DOWNSAMPLE_RATIO, dtype=np.int))
        img_mean = np.mean(img[chamber_limits_rs])
        # plt.imshow(img)
        # plt.scatter(x=chamber_limits_rs[1], y=chamber_limits_rs[0], c='red')

    if cell_mean > 1.8 * img_mean:  # these should be clear cells
        cell_type = 'cell'
    elif cell_mean > 1.50 * img_mean:  # cell 25%=40.96 / chamber 25%=26.88
        cell_type = 'cell_medium'
    return cell_type, cell_mean/img_mean


def calc_cell_class_thresh(cell, thresh_val=27):
    cell_type = 'cell_lite'
    cell_mean = np.mean(cell)
    if cell_mean > 1.8 * thresh_val:  # these should be clear cells
        cell_type = 'cell'
    elif cell_mean > 1.50 * thresh_val:  # cell 25%=40.96 / chamber 25%=26.88
        cell_type = 'cell_medium'
    return cell_type


# careful about flipping patches
def set_cell(img, cell, cell_coords, img_chamber_stats, cell_stats, augment=True, blur=True, augment_intensity=False, visualise=False):
    x_lower, y_lower, x_upper, y_upper = cell_coords
    # img[y_lower:y_upper, x_lower:x_upper] = cell
    old_cell = np.copy(cell)    # before any transformation/augmentation

    # random verticle/horizontal flip of cell
    if augment:
        if np.random.randint(0, 2) == 0:    # horizontal flip
            cell = cv2.flip(cell, 1)
        if np.random.randint(0, 2) == 0:    # vertical flip
            cell = cv2.flip(cell, 0)

        angle = np.random.choice([0, 90, 180, 270], 1)[0]
        if angle == 270:
            # cell = np.transpose(cell, (1, 0, 2))  # for 3D
            cell = np.transpose(cell)   # since 2D grayscale
            cell = cv2.flip(cell, 0)
        elif angle == 180:
            cell = cv2.flip(cell, -1)
        elif angle == 90:
            # cell = np.transpose(cell, (1, 0, 2))  # for 3D
            cell = np.transpose(cell)   # since 2D grayscale
            cell = cv2.flip(cell, 1)
        elif angle == 0:
            pass

    # adjust for intensity differences of patch chamber vs cell chamber
    if augment_intensity:
        (cell_img_imean, cell_img_istd, cell_imean, cell_istd, cell_chamber_imean, cell_chamber_istd) = cell_stats
        img_chamber_imean, img_chamber_istd, img_chamber_rows = img_chamber_stats
        adjust_factor = float(img_chamber_imean/cell_chamber_imean)
        adjust_factor *= np.random.uniform(low=0.8, high=1.2)
        cell = np.round(cell * adjust_factor)  # adjust cell intensity
        cell = np.clip(cell, 0., 255.)
        # FIXME - adjust for cell std vis-a-vis its chamber vs patch chamber??
        # something like cell2 = np.round((cell - cell_chamber_imean)/cell_chamber_istd * img_chamber_std + img_chamber_imean)

    cell_type, cell_brightness = calc_cell_class(cell, img)
    img[y_lower:y_upper, x_lower:x_upper] = cell
    if visualise:
        plt.figure(1)
        plt.clf()
        plt.imshow(cell)
        plt.title('intensity adjusted cell')
        plt.figure(2)
        plt.clf()
        plt.imshow(old_cell)
        plt.title('original cell')

    # smooth cell with background
    # img2 = cv2.GaussianBlur(img, (1, 1), 0) # this is do nothing
    # img2 = cv2.GaussianBlur(img, (3, 3), 0)
    # img3 = cv2.blur(img, (2, 2))
    # img2 = ndimage.gaussian_filter(img, sigma=.5)

    # only blur edges
    if blur:
        # img = my_blur_old(img, x_lower, y_lower, x_upper, y_upper, visualise=False) # cv2.blur is too smooth
        img3 = np.copy(img)
        img[y_lower - 2:y_lower + 1, x_lower:x_upper] = my_blur(img[y_lower-2:y_lower+1, x_lower:x_upper], dim=0)  # top
        img[y_upper:y_upper + 3, x_lower:x_upper] = my_blur(img[y_upper:y_upper + 3, x_lower:x_upper], dim=0)   # bottom
        img[y_lower:y_upper, x_lower - 2:x_lower + 1]= my_blur(img[y_lower:y_upper, x_lower-2:x_lower+1], dim=1)  # left
        img[y_lower:y_upper, x_upper:x_upper + 3] = my_blur(img[y_lower:y_upper, x_upper:x_upper + 3], dim=1)  # right

        if visualise:
            plt.figure(1)
            plt.clf()
            plt.imshow(img3)
            plt.title('original image with cell marked')
            plt.scatter(x=[x_lower, x_upper], y=[y_lower, y_upper], c='red', s=3)
            plt.figure(3)
            plt.clf()
            plt.imshow(img)
            plt.title('image with cell marked and blur')
            plt.scatter(x=[x_lower, x_upper], y=[y_lower, y_upper], c='red', s=3)
            smoothed_coords = np.argwhere(img != img3)
            plt.scatter(x=smoothed_coords[:, 1], y=smoothed_coords[:, 0], c='magenta', s=1)

    return img, cell_type


def my_blur_old(img, x_lower, y_lower, x_upper, y_upper, visualise=False):
    img3 = np.copy(img)
    # cv2.blur is too smooth
    img2 = cv2.blur(img, (2, 2))
    img[y_lower-1:y_lower+1, x_lower:x_upper] = img2[y_lower-1:y_lower, x_lower:x_upper]
    img[y_upper-1:y_upper+1, x_lower:x_upper] = img2[y_upper-1:y_upper+1, x_lower:x_upper]
    img[y_lower:y_upper, x_lower-1:x_lower+1] = img2[y_lower:y_upper, x_lower-1:x_lower+1]
    img[y_lower:y_upper, x_upper-1:x_upper+1] = img2[y_lower:y_upper, x_upper-1:x_upper+1]
    if visualise:
        plt.figure(1)
        plt.clf()
        plt.imshow(img3)
        plt.title('original image with cell marked')
        plt.scatter(x=[x_lower, x_upper], y=[y_lower, y_upper], c='red', s=3)
        plt.figure(2)
        plt.clf()
        plt.imshow(img2)
        plt.title('blurred image with cell marked')
        plt.scatter(x=[x_lower, x_upper], y=[y_lower, y_upper], c='red', s=3)
        plt.figure(3)
        plt.clf()
        plt.imshow(img)
        plt.title('image with cell marked and blur')
        plt.scatter(x=[x_lower, x_upper], y=[y_lower, y_upper], c='red', s=3)
        smoothed_coords = np.argwhere(img != img3)
        plt.scatter(x=smoothed_coords[:, 1], y=smoothed_coords[:, 0], c='yellow', s=1)
    return img


def my_blur(strip, std=None, dim=0):
    # strip should be ACCELL_DIAMETER*N
    if dim==1:
        strip = np.transpose(strip)

    strip_shape = strip.shape
    if strip_shape[0]==3:   # enough rows to interpolate
        if std==None:
            std = np.std(strip)
            adj_factor = 2  # more noise to be realistic?!
            adj_factor = 1
            std *= adj_factor
        strip[1, ] = np.random.normal((strip[0, ]+strip[2, ])/2, std)

    if dim==1:
        strip = np.transpose(strip)
    return np.around(strip, decimals=0)  # int array as images are integer-valued tensors


def check_predictions(raw_images, converted_images, img_names, img_preds, do_save=False):
    print(raw_images.shape, converted_images.shape, img_preds.shape)
    for idx, img_name in enumerate(img_names):
        cur_img = converted_images[idx, ]
        cur_pred = img_preds[idx, ]
        visualize_img_prediction(raw_images[idx, ], cur_img, cur_pred)
        # plt.figure(1)
        # plt.clf()
        # plt.subplot(131)
        # plt.imshow(cur_img)
        # plt.subplot(132)
        # plt.imshow(cur_pred)
        # plt.subplot(133)
        # plt.imshow(cur_img)
        # # non_zeros = np.transpose(np.nonzero(mask_pred))
        # pred_threshold = 0.9  # very conservative threshold
        # thresh_points = np.argwhere(cur_pred > pred_threshold)  # n*2 where either 1st/2nd coord below pred_threshold
        # plt.scatter(x=thresh_points[:, 1], y=thresh_points[:, 0], c='yellow', s=1)

        if do_save:
            save_folder = os.path.join(npy_data_folder, 'preds')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, img_name)
            plt.savefig(save_path)
    return


def get_larger_chambers(img_folder, accell_json_folder=segmentation_json_folder, do_avg=False):
    raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats \
        = get_scrubbed_imgs(img_folder, accell_json_folder=segmentation_json_folder, do_avg=do_avg)

    larger_idx = []
    larger_img_names = []
    for idx, img_name in enumerate(img_names):
        chamber_stat = chamber_stats[idx, ]
        chamber_imean, chamber_istd, chamber_rows = chamber_stat
        if chamber_rows > 15000:    # bigger chambers for sampling patches
            larger_idx.append(idx)
            larger_img_names.append(img_name)

    return raw_images[larger_idx, ], converted_imgs[larger_idx, ], larger_img_names, img_preds[larger_idx, ], \
           cleaned_imgs[larger_idx, ], chamber_stats[larger_idx, ]


# create training data of ac cells on background ac chamber patches
# get all labelled data
# get neighbouring slices
# convert neighbouring slices
# get predicted chambers for labelled data
# get ac cells for avg data
# split train/val by img
def create_accell_data_new(patch_rows=32, patch_cols=32, num_samples=10000, do_avg=True, do_blurred=False, visualise=False):
    # # get imgs, predictions and cleaned imgs without accell data
    # raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats = get_scrubbed_imgs(img_folder)   #'./accell/segmentations'
    # # get neighbouring slices
    # neighbor_names, npy_data = get_neighouring_pngs(img_names=img_names)

    # cell data with stats of img and chamber where they are from
    all_cell_data, all_cell_stats, cell_labels = get_all_ac_cells(segmentation_json_folder, img_folder, visualise=visualise, do_avg=do_avg)

    # only take cleaned_imgs with largest chambers for sampling patch
    raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats \
        = get_larger_chambers(img_folder, accell_json_folder=segmentation_json_folder, do_avg=do_avg)
    num_imgs = raw_images.shape[0]

    out_folder_name = 'ac_training{}{}_{}_{}'.format('_avg' if do_avg else '', '_blurred' if do_blurred else '', patch_rows, patch_cols)
    # out_folder_name = 'ac_training_avgNew_{}_{}'.format(patch_rows, patch_cols)

    if platform == "linux" or platform == "linux2":
        base_folder = '/data/yue/pepple/accell/{}'.format(out_folder_name)
    else:
        base_folder = './accell/{}'.format(out_folder_name)

    train_folder = '{}/train'.format(base_folder)
    valid_folder = '{}/valid'.format(base_folder)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        os.makedirs(train_folder)
        os.makedirs(valid_folder)

    # remove border (problematic) cells
    legit_indices, failed_indices = check_cells(all_cell_data, pixel_max=200, std_thresh=40)
    all_cell_data = all_cell_data[legit_indices, ]
    all_cell_stats = all_cell_stats[legit_indices, ]
    cell_labels = [x for idx, x in enumerate(cell_labels) if idx in legit_indices]

    # about 2500 cells - split appropriately;
    train_indices = [idx for idx, x in enumerate(cell_labels) if 'Kathryn' in x or 'Leslie' in x]
    cell_train = all_cell_data[train_indices, ]
    cell_train_stats = all_cell_stats[train_indices, ]
    create_accell_imgs(cleaned_imgs, converted_imgs, img_preds, chamber_stats, cell_train, cell_train_stats,
                       base_folder, num_samples, patch_rows, patch_cols, is_train=True, do_blurred=do_blurred, visualise=visualise)

    # validation data
    valid_indices = [idx for idx, x in enumerate(cell_labels) if 'Kathryn' not in x and 'Leslie' not in x]
    cell_valid = all_cell_data[valid_indices, ]
    cell_valid_stats = all_cell_stats[valid_indices, ]
    num_test = int(num_samples*.1)
    create_accell_imgs(cleaned_imgs, converted_imgs, img_preds, chamber_stats, cell_valid, cell_valid_stats,
                       base_folder, num_test, patch_rows, patch_cols, is_train=False, do_blurred=do_blurred, visualise=visualise)
    return


def check_cells(cells_data, pixel_max=200, std_thresh=50):
    num_cells = len(cells_data)
    failed_pix = []
    failed_std = []
    for idx, cur_cell in enumerate(cells_data):
        # intensity - any pixel in cell > threshold
        failed = False
        if np.any(cur_cell>pixel_max):
            failed_pix.append(idx)
            failed = True
        # std - too varied can also be signal for border
        if np.std(cur_cell)>std_thresh:
            failed_std.append(idx)
            failed = True
        if failed:
            plt.imshow(cur_cell)

    failed_indices = list(set().union(failed_pix, failed_std))
    legit_indices = list(set.difference(set(range(num_cells)), set(failed_indices)))    # probably inefficient
    return legit_indices, failed_indices


# .tiff names, but png data
def get_neighouring_pngs(img_names, conv_folder='converted_neighboring', num_to_avg=3):
    npy_file = 'neighbor_data.npy'
    neighbor_names_file = 'neighbors.txt'
    if os.path.isdir(conv_folder) and os.path.isfile(os.path.join(conv_folder, neighbor_names_file)):
        neighbor_names = []
        with open(os.path.join(conv_folder, neighbor_names_file), 'r') as fin:
            for l in fin.readlines():
                neighbor_names.append(l.rstrip())
        fin.close()

        npy_fpath = os.path.join(conv_folder, npy_file)
        if os.path.isfile(npy_fpath):
            npy_data = np.load(os.path.join(conv_folder, npy_file))
        else:
            npy_data = np.zeros(shape=(len(neighbor_names), RAW_IMG_ROWS, RAW_IMG_COLS))
            for idx, neighbor_name in enumerate(neighbor_names):
                npy_data[idx,] = cv2.imread(os.path.join(conv_folder, neighbor_name).replace('.TIFF', '.png'), cv2.IMREAD_GRAYSCALE)
            np.save(npy_fpath, npy_data)
    else:
        convert_folder = 'convert_neighboring'
        _, neighbor_names = copy_neighbouring(img_names, num_to_avg=num_to_avg, temp_folder=convert_folder)

        if not os.path.isdir(conv_folder):
            os.makedirs(conv_folder)
        with open(os.path.join(conv_folder, neighbor_names_file), 'w') as fout:
            for idx, neighbor_name in enumerate(neighbor_names):
                fout.write('{}\n'.format(neighbor_name))
        fout.close()

        # convert_neighboring(convert_folder, conv_folder)  # do on titan
    return neighbor_names, npy_data


# convert on titan to be consistent
def convert_neighboring(folder, out_folder):
    tiff_files = [x for x in os.listdir(folder) if '.tiff' in x.lower()]
    for idx, tiff_file in enumerate(tiff_files):
        convert_image(folder, tiff_file, out_folder, options='', debug=False)   # same scale for ac cells
    return


def copy_neighbouring(img_names, num_to_avg=3, temp_folder='convert_neighboring'):
    if not os.path.isdir(temp_folder):
        os.makedirs(temp_folder)

    from shutil import copyfile, copy2

    neighboring_names = []
    for idx, img_name in enumerate(img_names):
        # 'DeRuyter-Inflamed_20170703mouse1_Day2_Right_564'
        img_toks, img_base, img_num = parse_tiff_name(img_name, num_to_avg)
        img_nums = get_neighbouring_nums(img_num, num_to_avg)
        for jdx, num in enumerate(img_nums):
            img_name = '{} ({}).TIFF'.format(img_base, num)
            neighboring_names.append(img_name)
            src_path = os.path.join('accell', 'Inflamed', img_base, img_name)
            # cv2.imread(src_path).shape
            copy2(src_path, temp_folder)
    return temp_folder, neighboring_names


# avg and align images
def avg_images(img_names, num_to_avg=3, visualise=False):
    img_data = np.zeros(shape=(len(img_names), RAW_IMG_ROWS, RAW_IMG_COLS))

    neighbor_names, npy_data = get_neighouring_pngs(img_names=img_names)    # .tiff names, but png data
    # # check alignment of neighboring images
    # k = 10
    # temp = cv2.imread(os.path.join('converted_neighboring', neighbor_names[k].replace('.TIFF', '.png')),
    #                   cv2.IMREAD_GRAYSCALE)
    # np.sum(temp == npy_data[k,])

    for idx, img_name in enumerate(img_names):
        img_toks, img_base, img_num = parse_tiff_name(img_name, num_to_avg)
        img_nums = get_neighbouring_nums(img_num, num_to_avg)
        n_names = ['{} ({}).TIFF'.format(img_base, x) for x in img_nums]
        n_idx = [neighbor_names.index(x) for x in n_names]
        if len(n_idx)!=num_to_avg:
            print('something wrong')
        temp = npy_data[n_idx, ]
        img_data[idx, ] = np.mean(temp, axis=0)
        if visualise:
            plt.figure(1)
            temp2 = np.mean(temp, axis=0)
            plt.imshow(temp2)
            plt.title('averaged image for {}'.format(img_name))
            print(get_patch_stats(temp2))
            plt.figure(2)
            plt.imshow(temp[0,])
            plt.title('single image for {}'.format(img_name))
            print(get_patch_stats(temp[0,]))

    return np.around(img_data, decimals=0), img_names    # img_data into array of ints


def parse_tiff_name(img_name, num_to_avg=3):
    img_toks = img_name.split('_')
    img_base = '_'.join(img_toks[1:-1])
    img_num = int(img_toks[-1].replace('.png', ''))
    return img_toks, img_base, img_num


def get_neighbouring_nums(num, num_to_avg=3):
    img_div = num/float(num_to_avg)
    img_whole = int(np.ceil(img_div)-1)
    img_nums = list(range(img_whole*num_to_avg+1, (img_whole+1)*num_to_avg+1))
    return img_nums


# in situ cells around labelled data instead of plunking cells into random ambient ac background
def create_accell_insitu(folder=img_folder, patch_rows=32, patch_cols=32, num_samples=20000, do_avg=True, visualise=False):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)  # raw_images->converted(scale 50%)->predicted
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)
    num_imgs = raw_images.shape[0]

    if do_avg:
        recentered_json_folder = os.path.join('accell', 'jsons_recentered')
    else:
        # recentered_json_folder = os.path.join('accell', 'jsons_recentered_1scan')
        recentered_json_folder = os.path.join('accell', 'jsons_recentered_1scan_3by3')

    cell_dict = {}
    for idx, img_name in enumerate(img_names):
        cur_coords = get_coords(os.path.join(recentered_json_folder, img_name.replace('.png', '.json')))
        cell_dict[img_name] = cur_coords

    # split training and validation
    train_images = []
    train_preds = []
    train_cells = {}
    valid_images = []
    valid_preds = []
    valid_cells = {}

    kathryn_leslie_img_names = [x for x in sorted(img_names) if 'Kathryn' in x or 'Leslie' in x]
    num_kathryn_leslie = len(kathryn_leslie_img_names)
    train_split = 0.85
    train_end = int(np.floor(num_kathryn_leslie*train_split))
    train_names = kathryn_leslie_img_names[:train_end]
    valid_names = kathryn_leslie_img_names[train_end:]
    for img_name in train_names:
        train_idx = img_names.index(img_name)
        train_images.append(raw_images[train_idx,])
        train_preds.append(img_preds[train_idx,])
        train_cells[img_name] = cell_dict[img_name]

    for img_name in valid_names:
        valid_idx = img_names.index(img_name)
        valid_images.append(raw_images[valid_idx, ])
        valid_preds.append(img_preds[valid_idx, ])
        valid_cells[img_name] = cell_dict[img_name]

    train_images = np.asarray(train_images)
    valid_images = np.asarray(valid_images)
    train_preds = np.asarray(train_preds)
    valid_preds = np.asarray(valid_preds)

    # output folders
    # out_folder_name = 'ac_training_avg_insitu_{}_{}'.format(patch_rows, patch_cols)
    out_folder_name = 'ac_training_avg_insitu{}_{}_{}'.format('M' if MULTI_CLASS else '', patch_rows, patch_cols)
    out_folder_name = 'ac_training_avg_insitu{}_{}_{}'.format('nickHypo' if MULTI_CLASS else '', patch_rows, patch_cols)
    out_folder_name = 'ac_training_insitu{}'.format('_nickHypo' if MULTI_CLASS else '')  #  1-scan
    out_folder_name = 'ac_training_insitu{}_new'.format('_nickHypo' if MULTI_CLASS else '')  # 1-scan
    out_folder_name = 'ac_training_insitu{}_new2'.format('_nickHypo' if MULTI_CLASS else '')  # 1-scan, better boxes and brightness calc
    out_folder_name = 'ac_training_insitu{}_new3'.format('_nickHypo' if MULTI_CLASS else '')  # 1-scan, tight boxes and brightness calc
    base_folder = os.path.join(prefix, 'accell', format(out_folder_name))

    train_folder = '{}/train'.format(base_folder)
    valid_folder = '{}/valid'.format(base_folder)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        os.makedirs(train_folder)
        os.makedirs(valid_folder)

    # # remove border (problematic) cells
    # legit_indices, failed_indices = check_cells(all_cell_data, pixel_max=200, std_thresh=40)
    # all_cell_data = all_cell_data[legit_indices,]
    # all_cell_stats = all_cell_stats[legit_indices,]
    # cell_labels = [x for idx, x in enumerate(cell_labels) if idx in legit_indices]

    # about 2500 cells - split appropriately;
    create_accell_imgs_insitu(train_images, train_names, train_cells, patch_rows, patch_cols, num_samples=num_samples,
                              folder_name=train_folder, is_train=True, visualise=visualise, img_preds=train_preds)

    # validation data
    create_accell_imgs_insitu(valid_images, valid_names, valid_cells, patch_rows, patch_cols,
                              num_samples=int(num_samples*(1-train_split)), folder_name=valid_folder, is_train=False,
                              visualise=visualise, img_preds=valid_preds)
    return


## copied from human_accell/make_training_data.py
def find_cell_boundaries(img, cell_center, cell_size=ACCELL_DIAMETER, intensity_thresh=0.6, visualise=False):
    old_cell_coords = make_box_coords(cell_center, img.shape, box_size=ACCELL_DIAMETER)
    old_5by5_cell = get_cell_with_coords(img, old_cell_coords, tight=False)
    old_cell_coords = (old_cell_coords[0], old_cell_coords[1], old_cell_coords[2]-1, old_cell_coords[3]-1) # make everything tight for consistency

    img_shape = img.shape
    cur_x, cur_y = cell_center
    init_cell, init_cell_coords = get_cell(img, cur_x, cur_y, img_shape, cell_size=cell_size)

    # find brightest cells in initial boundary
    is_good_cell, img_coords, rel_coords = refine_cell(img, init_cell, init_cell_coords, intensity_thresh=intensity_thresh)
    good_cell = get_cell_with_coords(img, img_coords, tight=True)

    # new refine method
    new_cell_coords = refine_cell_new(img, cell_center, contrast_ratio=.75, last_expand_ratio=.9)
    new_cell = get_cell_with_coords(img, new_cell_coords, tight=True)

    if visualise:
        visualise_img_cell_boxes(img, cell_center, [init_cell_coords, img_coords, old_cell_coords, new_cell_coords])
        print('5*5 mean intensity={}; 11*11 mean intensity={}; max_intensity_thresh{}={}'
              .format(np.mean(old_5by5_cell), np.mean(init_cell), intensity_thresh, np.mean(good_cell), np.mean(new_cell)))
    return is_good_cell, img_coords, np.mean(good_cell), old_cell_coords, np.mean(old_5by5_cell), new_cell_coords, np.mean(new_cell)


def visualise_img_cell_boxes(img, cell_center, cell_coords, colors=['white', 'red', 'lime', 'blue'], size=2):
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    ax1.scatter(x=cell_center[0], y=cell_center[1], c='red', s=size)
    for idx, cell_coord in enumerate(cell_coords):
        x1, y1, x2, y2 = cell_coord
        color = colors[idx]
        ax1.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color))
    return


def refine_cell(img, init_cell, init_cell_coords, intensity_thresh=.3, visualise=False):
    x1, y1, x2, y2 = init_cell_coords
    # naive method of finding maximal intensity and thresholding largish patch/cell from maximal for cell
    # FIXME - dont necessarily maintain continuity
    max_intensity_ind = np.unravel_index(np.argmax(init_cell, axis=None), init_cell.shape)
    cell_max_intensity = init_cell[max_intensity_ind]   # ~100
    cell_median = np.median(init_cell)  # ~4/5
    # rel_intensity_thresh = min(cell_median*5, cell_max_intensity*intensity_thresh)
    rel_intensity_thresh = cell_max_intensity * intensity_thresh
    # high_intensity_pixels = np.nonzero(init_cell>cell_max_intensity*intensity_thresh)
    high_intensity_pixels = np.argwhere(init_cell > rel_intensity_thresh)  # p.argwhere flips x,y
    # relative coords to init_cell
    y1_new, x1_new, = np.min(high_intensity_pixels, axis=0)
    y2_new, x2_new = np.max(high_intensity_pixels, axis=0)
    x1_r, y1_r, x2_r, y2_r = x1_new + x1, y1_new + y1, x2_new + x1, y2_new + y1  # as image coordinates
    # new_cell = img[y1_r:y2_r+1, x1_r:x2_r+1]    # needs to include last points
    new_cell = get_cell_with_coords(img, (x1_r, y1_r, x2_r, y2_r), tight=True)
    good_cell, img_mean, cell_mean = True, np.mean(img), np.mean(new_cell)

    if visualise:
        cell_center = int((x1+x2)/2), int((y1+y2)/2)
        visualise_img_cell_boxes(img, cell_center, [init_cell_coords, (x1_r, y1_r, x2_r, y2_r)])
    return good_cell, (x1_r, y1_r, x2_r, y2_r), (x1_new, y1_new, x2_new, y2_new)
## end copied from human_accell/make_training_data.py


# iteratively refine border
def get_cell_with_coords(img, coords, tight=False):
    x1, y1, x2, y2 = coords
    if not tight:
        cell = img[y1:y2, x1:x2]
    else:
        cell = img[y1:y2+1, x1:x2+1]
    return cell


def refine_cell_new(img, cell_center, contrast_ratio=.8, last_expand_ratio=.9, visualise=False):
    x_center, y_center = cell_center

    # x1, y1 = x_center, y_center
    # x2, y2 = x_center, y_center
    # cell_center_intensity = img[y_center, x_center]
    x1, y1 = x_center-1, y_center-1     # 3*3 - to avoid being too flat
    x2, y2 = x_center+1, y_center+1
    cur_cell_coords = x1, y1, x2, y2
    # prev_cell_coords = cur_cell_coords
    prev_cell = get_cell_with_coords(img, cur_cell_coords, tight=True)
    cell_center_intensity = np.nanmean(prev_cell)
    prev_cell_mean = cell_center_intensity
    cur_cell_mean = prev_cell_mean

    counter = 0
    while cur_cell_mean>cell_center_intensity*contrast_ratio and cur_cell_mean>prev_cell_mean*last_expand_ratio and counter<10:
        counter += 1
        prev_cell_coords = cur_cell_coords  # update current to last
        prev_cell = get_cell_with_coords(img, prev_cell_coords, tight=True)
        prev_cell_mean = np.mean(prev_cell)

        x1, y1, x2, y2 = prev_cell_coords
        # check brightest direction
        up_coords = x1, y1-1, x2, y2
        down_coords = x1, y1, x2, y2+1
        left_coords = x1-1, y1, x2, y2
        right_coords = x1, y1, x2+1, y2
        direction_coords = [up_coords, down_coords, left_coords, right_coords]
        cell_contrast = []
        for direction_coord in direction_coords:
            direction_cell = get_cell_with_coords(img, direction_coord, tight=True)
            cell_contrast.append(np.nanmean(direction_cell))
        max_idx = np.argmax(cell_contrast)
        cur_cell_coords = direction_coords[max_idx]
        # cur_cell = get_cell_with_coords(img, cur_cell_coords, tight=True)
        cur_cell_mean = cell_contrast[max_idx]

        if visualise:
            cell_center = int((x1 + x2) / 2), int((y1 + y2) / 2)
            visualise_img_cell_boxes(img, cell_center, [cur_cell_coords])

    return prev_cell_coords


def create_accell_imgs_insitu(images, img_names, cell_dict, patch_rows, patch_cols, num_samples=20000, folder_name='',
                              is_train=True, visualise=False, img_preds=None):
    img_shape = images.shape
    num_images = img_shape[0]

    # weight according to cells in image
    # weights = np.ones((num_images, 1), dtype=np.float32)  # equal weighting images
    weights = []
    for idx, img_name in enumerate(img_names):
        weights.append(len(cell_dict[img_name])*1.0)
    print('create_accell_imgs_insitu', np.sum(weights), weights)
    weights = np.array(weights)/np.sum(weights)   # weighting images by cells
    print('create_accell_imgs_insitu', np.sum(weights), weights)

    for s_index in range(num_samples):
        print('processing', s_index)
        # get random index, thereby image and img_name
        r_idx = np.random.choice(num_images, p=weights)
        # aliasing
        cur_img = images[r_idx,]
        cur_name = img_names[r_idx]
        img_cells = cell_dict[cur_name]
        cur_pred = img_preds[r_idx, ]

        # sample random cell in img
        c_idx = np.random.choice(len(img_cells))
        cur_cell = img_cells[c_idx]

        # get random patch around cell; patch_cell_coord is relative to cell ie. [0,32]
        good_cell, cell_coords_refined, cell_mean, cell_coords_fixed, cell_mean_fixed, cell_coords_ref_new, cell_mean_ref_new = \
            find_cell_boundaries(cur_img, cur_cell, cell_size=ACCELL_DIAMETER * 2, intensity_thresh=0.7)
        x1 = [cell_coords_refined[0], cell_coords_fixed[0], cell_coords_ref_new[0]]
        x2 = [cell_coords_refined[2], cell_coords_fixed[2], cell_coords_ref_new[2]]
        y1 = [cell_coords_refined[1], cell_coords_fixed[1], cell_coords_ref_new[1]]
        y2 = [cell_coords_refined[3], cell_coords_fixed[3], cell_coords_ref_new[3]]
        max_cell_coord = [np.min(x1), np.min(y1), np.max(x2), np.max(y2)]   # maximum coords for patch

        # # record cell means
        # out_name = '{}_{}{}.png'.format(cur_name, 'train' if is_train else 'valid', s_index)
        # with open('cell_construction_test_file.csv', 'a') as fout:
        #     vals = [out_name] + list(cell_coords_refined) + [cell_mean] + list(cell_coords_fixed) + \
        #            [cell_mean_fixed] + list(cell_coords_ref_new) + [cell_mean_ref_new]
        #     fout.write('{}\n'.format(','.join([str(x) for x in vals])))
        # fout.close()

        # check if it includes other cells; patch_coords are relative to image
        # cur_patch, patch_cell_coord, real_x, real_y = get_patch_for_cell(cur_img, cur_cell, patch_rows, patch_cols)
        cur_patch, patch_cell_coord, real_x, real_y = get_patch_for_cell(cur_img, max_cell_coord, patch_rows, patch_cols)
        # patch_coords = [real_x, real_y, real_x + patch_cols - 1, real_y + patch_rows - 1]
        patch_coords = [real_x, real_y, real_x + patch_cols, real_y + patch_rows]

        overlapping_coords_fixed = []
        overlapping_coords_new = []
        overlapping_coords_old = []
        for cell_coord in img_cells:    # check if other img_cells in this patch
            good_cell, cell_coords_refined, cell_mean, cell_coords_fixed, cell_mean_fixed, cell_coords_ref_new, cell_mean_ref_new = \
                find_cell_boundaries(cur_img, cell_coord, cell_size=ACCELL_DIAMETER * 2, intensity_thresh=0.7)   # get best bounding box

            target_coords = cell_coords_refined
            if overlap(target_coords, patch_coords):  # overlap using coords relative to img
                overlap_coords = make_overlapping_box(target_coords, patch_coords)
                if _significant_overlap(patch_coords, target_coords, overlap_coords):
                    cur_cell = get_cell_with_coords(cur_img, target_coords, tight=True)  # find_cell_boundaries is tight
                    # use cell predictions for better brightness - otherwise img_pred=None
                    cell_type, cell_brightness = calc_cell_class(cur_cell, cur_img, img_pred=cur_pred)  # against img for more robustness; patch can be on ac boundary
                    # overlapping_coords.append(tuple(list(adj_coords(overlap_coords, real_x, real_y)) + [cell_type]))  # only parts that are in image
                    overlapping_coords_old.append(tuple(list(adj_coords(overlap_coords, real_x, real_y)) + [cell_brightness]))

            target_coords = cell_coords_fixed
            if overlap(target_coords, patch_coords):  # overlap using coords relative to img
                overlap_coords = make_overlapping_box(target_coords, patch_coords)
                if _significant_overlap(patch_coords, target_coords, overlap_coords):
                    cur_cell = get_cell_with_coords(cur_img, target_coords, tight=True)  # find_cell_boundaries is tight
                    # use cell predictions for better brightness - otherwise img_pred=None
                    cell_type, cell_brightness = calc_cell_class(cur_cell, cur_img, img_pred=cur_pred)  # against img for more robustness; patch can be on ac boundary
                    overlapping_coords_fixed.append(tuple(list(adj_coords(overlap_coords, real_x, real_y)) + [cell_brightness]))

            target_coords = cell_coords_ref_new
            if overlap(target_coords, patch_coords):  # overlap using coords relative to img
                overlap_coords = make_overlapping_box(target_coords, patch_coords)
                if _significant_overlap(patch_coords, target_coords, overlap_coords):
                    cur_cell = get_cell_with_coords(cur_img, target_coords, tight=True)  # find_cell_boundaries is tight
                    # use cell predictions for better brightness - otherwise img_pred=None
                    cell_type, cell_brightness = calc_cell_class(cur_cell, cur_img, img_pred=cur_pred)  # against img for more robustness; patch can be on ac boundary
                    overlapping_coords_new.append(tuple(list(adj_coords(overlap_coords, real_x, real_y)) + [cell_brightness]))

        if visualise:
            plt.figure(1)   # original
            plt.clf()
            plt.imshow(cur_img)
            plt.axes().add_patch(patches.Rectangle((real_x, real_y), patch_cols, patch_rows, fill=False, color='green'))
            for coord in img_cells:
                plt.scatter(x=coord[0], y=coord[1], c='red', s=1)   # raw coords

            np.sum(cur_img[patch_coords[1]:patch_coords[3]+1, patch_coords[0]:patch_coords[2]+1] == cur_patch)
            plt.figure(2)   # patch
            plt.clf()
            plt.imshow(cur_patch)
            for coord in overlapping_coords_new:
                plt.scatter(x=[coord[0], coord[0], coord[2], coord[2]], y=[coord[1], coord[1], coord[3], coord[3]], c='red', s=1)
                plt.axes().add_patch(patches.Rectangle((coord[0], coord[1]), coord[2]-coord[0], coord[3]-coord[1], fill=False, color='red'))

        # write image
        out_name = '{}_{}.png'.format('train' if is_train else 'valid', s_index)
        cv2.imwrite(os.path.join(folder_name, out_name), cur_patch)
        # write coords
        with open(os.path.join(folder_name, 'train_coords_fixed.txt' if is_train else 'valid_coords_fixed.txt'), 'a') as fout:
            for jdx, coord in enumerate(overlapping_coords_fixed):
                if MULTI_CLASS:
                    vals = [out_name] + [str(x) for x in coord]
                else:
                    vals = [out_name] + [str(x) for x in coord[:-1]] + ['cell']
                fout.write('{}\n'.format(','.join(vals)))
        fout.close()

        with open(os.path.join(folder_name, 'train_coords_old.txt' if is_train else 'valid_coords_old.txt'), 'a') as fout:
            for jdx, coord in enumerate(overlapping_coords_old):
                if MULTI_CLASS:
                    vals = [out_name] + [str(x) for x in coord]
                else:
                    vals = [out_name] + [str(x) for x in coord[:-1]] + ['cell']
                fout.write('{}\n'.format(','.join(vals)))
        fout.close()

        with open(os.path.join(folder_name, 'train_coords_new.txt' if is_train else 'valid_coords_new.txt'), 'a') as fout:
            for jdx, coord in enumerate(overlapping_coords_new):
                if MULTI_CLASS:
                    vals = [out_name] + [str(x) for x in coord]
                else:
                    vals = [out_name] + [str(x) for x in coord[:-1]] + ['cell']
                fout.write('{}\n'.format(','.join(vals)))
        fout.close()

    return s_index


def adj_coords(coord, rand_x, rand_y, num_rows=32, num_cols=32):
    x1, y1, x2, y2 = coord
    # pix_buffer = 2
    pix_buffer = 0
    adj_x1 = max(0, x1 - rand_x + pix_buffer)
    adj_x2 = min(x2 - rand_x - pix_buffer, num_cols-1)
    adj_y1 = max(0, y1 - rand_y + pix_buffer)
    adj_y2 = min(y2 - rand_y - pix_buffer, num_rows-1)
    return (adj_x1, adj_y1, adj_x2, adj_y2)


def get_patch_for_cell(img, cell_coord, patch_row, patch_col, img_rows=1024, img_cols=1000, visualise=False):
    pix_buffer = 0
    if len(cell_coord)==2:
        x, y = cell_coord
        # TODO - use code from human_accell/make_training_data.py/find_cell_boundaries(img, img_cell, cell_size=ACCELL_DIAMETER*2)
        x1, y1, x2, y2 = make_box_coords((x, y), img.shape, box_size=ACCELL_DIAMETER)   # img_shape in case on edge of large img
    else:
        x1, y1, x2, y2 = cell_coord

    img_rows, img_cols = img.shape
    x_range = [max(int(x2 - patch_col + pix_buffer), 0), min(x1 - pix_buffer, img_cols - patch_col)]
    y_range = [max(int(y2 - patch_row + pix_buffer), 0), min(y1 - pix_buffer, img_rows - patch_row)]
    rand_x = random.randint(min(x_range[0], x_range[1]), max(x_range[0], x_range[1]))  # furtherest east and west points
    rand_y = random.randint(min(y_range[0], y_range[1]), max(y_range[0], y_range[1]))  # furtherest north and south points

    # grab patch around cell and adjust coord appropriately
    rand_patch = img[rand_y:rand_y + patch_row, rand_x:rand_x + patch_col]
    adjusted_coord = adj_coords((x1, y1, x2, y2), rand_x, rand_y, num_rows=patch_row, num_cols=patch_col)

    if rand_patch.shape != (patch_row, patch_col):
        print(rand_patch.shape)

    if visualise:
        plt.figure(200)
        plt.imshow(rand_patch)
        adj_x1, adj_y1, adj_x2, adj_y2 = adjusted_coord
        plt.scatter(x=[adj_x1, adj_x1, adj_x2, adj_x2], y=[adj_y1, adj_y2, adj_y1, adj_y2], c='red', s=2)
    return rand_patch, adjusted_coord, rand_x, rand_y


def make_overlapping_box(box_coord_1, box_coord_anchor):
    x1_low, y1_low, x1_upper, y1_upper = box_coord_1
    x2_low, y2_low, x2_upper, y2_upper = box_coord_anchor

    overlap_x1 = max([x1_low, x2_low])
    overlap_x2 = min([x1_upper, x2_upper])
    overlap_y1 = max([y1_low, y2_low])
    overlap_y2 = min([y1_upper, y2_upper])
    if overlap_x1>overlap_x2:
        overlap_x1, overlap_x2 = -1, -2    # not real overlap
    if overlap_y1>overlap_y2:
        overlap_y1, overlap_y2 = -1, -2     # not real overlap
    return (overlap_x1, overlap_y1, overlap_x2, overlap_y2)


# Overlapping rectangles overlap both horizontally & vertically
def overlap(coord_1, coord_2):
    x_lower_1, y_lower_1, x_upper_1, y_upper_1 = coord_1
    x_lower_2, y_lower_2, x_upper_2, y_upper_2 = coord_2
    x_overlaps = range_overlap(x_lower_1, x_upper_1, x_lower_2, x_upper_2)
    y_overlaps = range_overlap(y_lower_1, y_upper_1, y_lower_2, y_upper_2)
    return x_overlaps and y_overlaps


# Neither range is completely greater than the other
def range_overlap(a_min, a_max, b_min, b_max):
    return (a_min <= b_max) and (b_min <= a_max)


def _box_area(box_coords):
    x1, y1, x2, y2 = box_coords
    if x1>x2 or y1>y2 or np.any(np.array(box_coords)<0):
        print('bad coords', box_coords)
        return 0
    # return float((x2-x1)*(y2-y1))
    return float((x2 - x1 + 1) * (y2 - y1 + 1))


def _significant_overlap(box_anchor_coords, cell_coords, overlap_coords):
    cell_area = _box_area(cell_coords)
    overlap_area = _box_area(overlap_coords)
    x1, y1, x2, y2 = overlap_coords
    x_range = x2 - x1
    y_range = y2 - y1
    # conditions: bad coords, too small, too elongated
    if (x_range <= 0 or y_range <= 0) or overlap_area/cell_area < 0.75 or (x_range/y_range > 5 or y_range/x_range > 5):
        return False
    else:
        return True


def extreme_intensity_img(img_names, num_to_avg=3, get_max=True):
    img_data = np.zeros(shape=(len(img_names), RAW_IMG_ROWS, RAW_IMG_COLS))

    neighbor_names, npy_data = get_neighouring_pngs(img_names=img_names)  # .tiff names, but png data

    for idx, img_name in enumerate(img_names):
        img_toks, img_base, img_num = parse_tiff_name(img_name, num_to_avg)
        img_nums = get_neighbouring_nums(img_num, num_to_avg)
        n_names = ['{} ({}).TIFF'.format(img_base, x) for x in img_nums]
        n_idx = [neighbor_names.index(x) for x in n_names]
        if len(n_idx) != num_to_avg:
            print('something wrong')
        temp = npy_data[n_idx,]
        if get_max:
            img_data[idx,] = np.max(temp, axis=0)
        else:
            img_data[idx,] = np.min(temp, axis=0)
    return img_data, img_names


def review_recentered_coords(overplot=True):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(img_folder)
    averaged_images, _ = avg_images(img_names)
    max_images, _ = extreme_intensity_img(img_names, get_max=True)
    min_images, _ = extreme_intensity_img(img_names, get_max=False)

    recentered_json_folder = os.path.join('accell', 'jsons_recentered')
    cell_dict = {}
    for idx, img_name in enumerate(img_names):
        cur_img = raw_images[idx, ]
        cur_img_avg = averaged_images[idx, ]

        coords_manual = get_coords(os.path.join(segmentation_json_folder, img_name.replace('.png', '.json')))
        coords_recentered = get_coords(os.path.join(recentered_json_folder, img_name.replace('.png', '.json')))

        plt.figure(1)
        plt.clf()
        plt.imshow(cur_img)
        plt.title('1-scan: img{} {}'.format(idx, img_name))
        for cell in coords_manual:
            x, y = cell
            plt.scatter(x=[x], y=[y], c='red', s=1)
        if overplot:
            for cell in coords_recentered:
                x, y = cell
                plt.scatter(x=[x], y=[y], c='lime', s=1)

        plt.figure(2)
        plt.clf()
        plt.imshow(cur_img_avg)
        plt.title('averaged-scan: img{} {}'.format(idx, img_name))
        for cell in coords_recentered:
            x, y = cell
            plt.scatter(x=[x], y=[y], c='lime', s=1)
        if overplot:
            for cell in coords_manual:  # overplot manual
                x, y = cell
                plt.scatter(x=[x], y=[y], c='red', s=1)

        plt.figure(3)
        plt.clf()
        plt.imshow(max_images[idx, ])
        plt.title('max-scan: img{} {}'.format(idx, img_name))
        for cell in coords_recentered:
            x, y = cell
            plt.scatter(x=[x], y=[y], c='lime', s=1)
        if overplot:
            for cell in coords_manual:  # overplot manual
                x, y = cell
                plt.scatter(x=[x], y=[y], c='red', s=1)

        plt.figure(4)
        plt.clf()
        plt.imshow(min_images[idx, ])
        plt.title('min-scan: img{} {}'.format(idx, img_name))
        for cell in coords_recentered:
            x, y = cell
            plt.scatter(x=[x], y=[y], c='lime', s=1)
        if overplot:
            for cell in coords_manual:  # overplot manual
                x, y = cell
                plt.scatter(x=[x], y=[y], c='red', s=1)

        print(idx, img_name, img_names[idx])
    return


def zoom_experiment(zoom_size=600, do_avg=True):
    # raw_images, converted_imgs, img_names, img_preds = get_img_predictions(img_folder)
    # averaged_images, _ = avg_images(img_names)
    # max_images, _ = extreme_intensity_img(img_names, get_max=True)
    # min_images, _ = extreme_intensity_img(img_names, get_max=False)

    if do_avg:
        img_folder = os.path.join('accell', 'ac_training_avg_insituM_32_32', 'valid')
    else:
        img_folder = os.path.join('accell', 'ac_training3_32_32', 'valid')
    img_names = [x for x in os.listdir(img_folder) if '.png' in x]

    from analyseCellPreds import get_true_coords_file
    true_dict = get_true_coords_file(img_folder.replace('valid', ''), coord_file='valid_coords.txt',
                                     path_prefix='/data/yue/pepple/accell/ac_training_avg_insituM_32_32/valid/')

    img_min_side = float(zoom_size)
    for idx, img_name in enumerate(img_names):
        cur_img = cv2.imread(os.path.join(img_folder, img_name), cv2.IMREAD_GRAYSCALE)
        # cur_img = raw_images[idx, ]
        # cur_img_avg = averaged_images[idx, ]
        img_shapes = cur_img.shape
        height, width = img_shapes[0], img_shapes[1]

        if width <= height:
            f = img_min_side / width
            new_height = int(f * height)
            new_width = int(img_min_side)
        else:
            f = img_min_side / height
            new_width = int(f * width)
            new_height = int(img_min_side)
        fx = width / float(new_width)
        fy = height / float(new_height)

        zoomed_img = cv2.resize(cur_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        img_cells = true_dict[img_name]
        plt.figure(1)
        plt.clf()
        plt.imshow(cur_img)
        plt.title('{}-scan: {} {}'.format('3' if do_avg else '1', idx, img_name))
        for cell in img_cells:
            x1, y1, x2, y2 = cell
            plt.scatter(x=[x1, x1, x2, x2], y=[y1, y2, y1, y2], c='red', s=1)
        plt.figure(3)
        plt.clf()
        plt.imshow(zoomed_img)
        plt.title('{}-zoomed: {} {}'.format('3' if do_avg else '1', idx, img_name))
        for cell in img_cells:
            x1, y1, x2, y2 = cell
            plt.scatter(x=[x1/fx, x1/fx, x2/fx, x2/fx], y=[y1/fy, y2/fy, y1/fy, y2/fy], c='red', s=1)

        print(idx, img_name)
    return


def keep_classes(folder, coord_file, remove_classes=[]):
    lines = []
    with open(os.path.join(folder, coord_file), 'r') as fin:
        for l in fin.readlines():
            lines.append(l.rstrip())
    fin.close()

    with open(os.path.join(folder, 'train_coords_class.txt'), 'w') as fout:
        for l in lines:
            remove = False
            for cls in remove_classes:
                remove = cls in l or remove
            if not remove:
                fout.write('{}\n'.format(l))
    fout.close()
    return


# for making difrerent thresholded cell types
def make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'train', 'train_coords.txt')):
    # outfile = file_path.replace('_coords.txt', '_coords_{}.txt'.format(thresh_level))
    # outfile = file_path.replace('_coords_new.txt', '_coords_new_{}.txt'.format(thresh_level))
    outfile = file_path.replace('_coords_fixed.txt', '_coords_fixed_{}.txt'.format(thresh_level))
    with open(outfile, 'w') as fout:
        with open(file_path, 'r') as fin:
            for l in fin.readlines():
                l_toks = l.rstrip().split(',')
                x1, y1, x2, y2 = [int(x) for x in l_toks[1:-1]]
                cell_brightness = float(l_toks[-1])
                if cell_brightness>thresh_level and x1<x2 and y1<y2:
                    fout.write(l.replace(l_toks[-1], 'cell'))
        fin.close()
    fout.close()
    return


def check_created_accell_data(filepath):
    data = []
    with open(filepath, 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            data.append([float(x) for x in l_toks[1:]])
    fin.close()

    # analyse data
    data = np.array(data)
    # [out_name] + list(img_cell_coords) + [cell_mean] + list(old_cell_coords) + [old_cell_mean] + list(new_cell_coords) + [new_cell_mean]
    # compare cell_means
    old_refine_cell_mean = data[:, 5-1]
    fixed_5by5_cell_mean = data[:, 10-1]
    new_refine_cell_mean = data[:, 15-1]
    plt.figure(1)
    plt.clf()
    plt.hist([old_refine_cell_mean, fixed_5by5_cell_mean, new_refine_cell_mean])
    plt.legend(['old_refine', 'fixed_5by5', 'new_refine'])
    # paired analysis is more useful
    plt.figure(2)
    plt.clf()
    new_minus_fixed = new_refine_cell_mean-fixed_5by5_cell_mean
    new_minus_old = new_refine_cell_mean-old_refine_cell_mean
    old_minus_fixed = old_refine_cell_mean-fixed_5by5_cell_mean
    plt.plot(np.transpose([new_minus_fixed, new_minus_old, old_minus_fixed]), 'o')
    plt.legend(['new_minus_fixed', 'new_minus_old', 'old_minus_fixed'])
    probs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    print(np.percentile(new_minus_fixed, q=probs))
    print(np.percentile(new_minus_old, q=probs))
    print(np.percentile(old_minus_fixed, q=probs))

    # compare cell size
    old_refine_cell_area = (data[:, 3-1] - data[:, 1-1] +1)*(data[:, 4-1] - data[:, 2-1] +1)
    fixed_5by5_cell_area = (data[:, 8-1] - data[:, 6-1] )*(data[:, 9-1] - data[:, 7-1] )
    new_refine_cell_area = (data[:, 13-1] - data[:, 11-1] +1)*(data[:, 14-1] - data[:, 12-1] +1)
    plt.figure(3)   # compare old vs new refine
    plt.clf()
    plt.plot(new_refine_cell_area, old_refine_cell_area, 'o')
    ax = plt.axes()
    x_lim = ax.get_xlim()
    x = np.linspace(*ax.get_xlim())
    plt.plot(x, x)

    plt.figure(4)
    plt.clf()
    plt.hist([old_refine_cell_area, fixed_5by5_cell_area, new_refine_cell_area])
    plt.legend(['old_refine', 'fixed_5by5', 'new_refine'])

    plt.figure(5)
    plt.clf()
    plt.plot(np.transpose([old_refine_cell_area, fixed_5by5_cell_area, new_refine_cell_area]), np.transpose([new_minus_fixed, new_minus_old, old_minus_fixed]), 'o')
    plt.legend(['old_refine', 'fixed_5by5', 'new_refine'])

    return


if __name__ == '__main__':
    # predict all inflamed ac chambers
    # raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder=img_folder)
    # for img_name in img_names:
    #     print(img_name.split('-')[-1])

    # temp = cv2.resize(raw_images[0,], (2024, 2000), interpolation=cv2.INTER_CUBIC)
    # plt.figure(1)
    # plt.imshow(raw_images[0,])
    # plt.figure(2)
    # plt.imshow(temp)
    # print(1)

    # # check_predictions(raw_images, converted_imgs, img_names, img_preds, do_save=True)
    #
    # # clean raw_images - remove segmented ac cells from chambers by resampling chamber
    # raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats \
    #     = get_scrubbed_imgs(img_folder, accell_json_folder=segmentation_json_folder)
    #
    # # only take cleaned_imgs with largist chambers for sampling patch
    # raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats \
    #     = get_larger_chambers(img_folder, accell_json_folder=segmentation_json_folder)
    #
    # grab and store all ac cell data + their chamber intensity mean and std
    # all_cell_data, all_cell_stats, cell_labels \
    #     = get_all_ac_cells(seg_folder=segmentation_json_folder, img_folder=img_folder, visualise=False)
    #
    # # patch_sampler - check for erroneous/edge patches and too high intensity
    # # plunk cells in patch from patch_sampler - adjust for relative intensity of original chamber vs patch chamber
    # create_accell_imgs(cleaned_imgs, converted_imgs, img_preds, chamber_stats, all_cell_data, all_cell_stats,
    #                    output_folder='./accell/new_32_32/',
    #                    num_samples=10000, patch_rows=32, patch_cols=32, is_train=True, visualise=False)

    # # create training and validation data
    # create_accell_data(patch_rows=32, patch_cols=32, num_samples=20000, visualise=False)

    # using 3 avg OCTs
    # img_names = [x for x in os.listdir(img_folder) if '.png' in x and 'mask' not in x]
    # neighbor_names, npy_data = get_neighouring_pngs(img_names=img_names)
    # create_accell_data_new(patch_rows=32, patch_cols=32, num_samples=20000, visualise=False)
    # create_accell_data_new(patch_rows=32, patch_cols=32, num_samples=20000, visualise=False, do_blurred=True)

    # keep 2 classes
    # keep_classes(folder=os.path.join('accell', 'ac_training_avg_insitunickHypo_32_32', 'train'),
    #              coord_file='train_coords.txt', remove_classes=['cell_lite'])
    # keep_classes(folder=os.path.join('accell', 'ac_training_avg_blurred_32_32', 'train'),
    #              coord_file='training_coords.txt', remove_classes=['cell_lite'])
    # keep_classes(folder=os.path.join('accell', 'ac_training_insitu_nickHypo', 'train'),
    #              coord_file='train_coords.txt', remove_classes=['cell_lite'])

    # # test zooming of images, which happens in keras-frcnn-master
    # zoom_experiment(zoom_size=600)

    # # review recentered coords
    # review_recentered_coords()

    # # better focused 3*3 cells for expansion
    # raw_images, converted_imgs, img_names, img_preds = get_img_preds_wrapper(img_folder, do_avg=False, visualise=False)
    # recenter_cells_for_images(raw_images, img_names, img_preds, segmentation_json_folder, cell_size=3, do_avg=False)

    MULTI_CLASS = True  # allow 3 classes
    create_accell_insitu(folder=img_folder, patch_rows=32, patch_cols=32, num_samples=20000, visualise=False, do_avg=False)
    check_created_accell_data(filepath=os.path.join('cell_construction_test_file.csv'))

    # make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.75, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'train', 'train_coords.txt'))

    # make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.75, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new', 'valid', 'valid_coords.txt'))

    # make_cell_classes_by_thresh_level(thresh_level=1.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.75, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'train', 'train_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'train', 'train_coords.txt'))
    #
    # make_cell_classes_by_thresh_level(thresh_level=1.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.75, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid', 'valid_coords.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid', 'valid_coords.txt'))

    # # better adaptive cells
    # make_cell_classes_by_thresh_level(thresh_level=1.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.75, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_new.txt'))
    #
    # make_cell_classes_by_thresh_level(thresh_level=1.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=1.75, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_new.txt'))
    # make_cell_classes_by_thresh_level(thresh_level=2.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_new.txt'))

    # fixed 5*5 cells for new data 'ac_training_insitu_nickHypo_new3'
    make_cell_classes_by_thresh_level(thresh_level=1.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=1.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=1.75, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=2.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=2.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=2.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'train', 'train_coords_fixed.txt'))

    make_cell_classes_by_thresh_level(thresh_level=1.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=1.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=1.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=1.75, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=2.0, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=2.25, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_fixed.txt'))
    make_cell_classes_by_thresh_level(thresh_level=2.5, file_path=os.path.join(prefix, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_fixed.txt'))

    # # json for recentered cells for 1-scan or 3-avg
    # do_avg = False
    # all_cell_data, all_cell_stats, cell_labels \
    #     = get_all_ac_cells(seg_folder=segmentation_json_folder, img_folder=img_folder, visualise=False, do_avg=do_avg)
    # create_accell_insitu(folder=img_folder, patch_rows=32, patch_cols=32, num_samples=20000, do_avg=do_avg, visualise=False)