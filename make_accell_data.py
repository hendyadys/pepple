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
    segmentation_json_folder = '/home/ayl/data/pepple/accell/jsons'
    img_folder = '/home/ayl/data/pepple/accell/segmentations'
    empty_img_folder = '/home/ayl/data/pepple/accell/empty_segmentations'

    npy_data_folder = '/home/yue/pepple/accell/npy_data'
    # chamber_weights_folder = '/home/yue/pepple/runs/2017-08-09-10-20-24/weights'

    # includes more (unique) images
    chamber_weights_folder = './runs/2017-12-12-17-00-53'
    # test_weights = 'weights-improvement-174--0.83855137.hdf5'
    # test_weights = 'weights-improvement-256--0.85407762.hdf5'
elif platform == "win32":
    # Windows...
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


# load raw images (1024*1000)
def get_raw_imgs(folder):
    img_names = [x for x in sorted(os.listdir(folder)) if '.png' in x.lower() and 'mask' not in x.lower()]
    output_npy = os.path.join(npy_data_folder, 'raw_images.npy')
    if os.path.isfile(output_npy):
        raw_images = np.load(output_npy)
        return raw_images, img_names

    num_images = len(img_names)
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
def convert_raw_imgs(folder):
    raw_images, img_names = get_raw_imgs(folder)
    converted_npy = os.path.join(npy_data_folder, 'converted_imgs.npy')
    if os.path.isfile(converted_npy):
        converted_imgs = np.load(converted_npy)
        return raw_images, converted_imgs, img_names

    num_images = raw_images.shape[0]
    converted_imgs = np.ndarray((num_images, SCALED_IMG_ROWS, SCALED_IMG_COLS), dtype=np.float32)
    output_dir = './accell/seg_scaled'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, img_name in enumerate(img_names):
        converted_img_name = convert_image(folder, img_name, output_dir)
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
def get_img_predictions(folder=img_folder):
    raw_images, converted_imgs, img_names = convert_raw_imgs(folder)
    pred_npy = os.path.join(npy_data_folder, 'img_preds.npy')
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
    return raw_images, converted_imgs, img_names, img_preds


def read_coords(coord_file):
    fin = open(coord_file).read()
    json_data = json.loads(fin)
    return json_data


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
    img_height = img_shape[0]
    img_width = img_shape[1]

    lower_x = max(cur_x - (box_size // 2), 0)
    upper_x = min(cur_x + (box_size // 2)+1, img_width)

    lower_y = max(cur_y - (box_size // 2), 0)
    upper_y = min(cur_y + (box_size // 2)+1, img_height)
    return lower_x, lower_y, upper_x, upper_y


# from predicted images: 1. rescale, 2. remove segmented cells 3. by resampling segmented ac chamber, 4. store stats
def get_scrubbed_imgs(folder, accell_json_folder=segmentation_json_folder):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)
    cleaned_npy = os.path.join(npy_data_folder, 'cleaned_imgs.npy')
    chamber_stats_npy = os.path.join(npy_data_folder, 'chamber_stats.npy')
    if os.path.isfile(cleaned_npy) and os.path.isfile(chamber_stats_npy):
        cleaned_imgs = np.load(cleaned_npy)
        chamber_stats = np.load(chamber_stats_npy)  # can get img stats directly - so no need to compute here
        return raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats

    num_images, img_rows, img_cols = converted_imgs.shape   # 512*500
    raw_images_old = np.copy(raw_images)
    chamber_stats = np.ndarray((num_images, 3), dtype=np.float32)
    for idx, img_name in enumerate(img_names):
        print('scrubbing {}'.format(img_name))
        # get corresponding ac coords for img
        coord_file = os.path.join(accell_json_folder, img_name.replace('.png', '.json'))
        coords = get_coords(coord_file)
        cur_img = raw_images[idx, ]     # want cleaned_imgs on original scale (1024*1000)
        img_shape = cur_img.shape
        img_pred = img_preds[idx, ]     # predictions are 512*512
        scaled_img = converted_imgs[idx, ]
        for coord in coords:    # scrub each coord
            # get a patch in chamber to replace segmented accell
            rand_patch, patch_coords, scaled_coords, chamber_mean, chamber_std, chamber_shape = \
                img_sampler(cur_img, scaled_img, img_pred, patch_size=ACCELL_DIAMETER, centralize_factor=1, all_in=True)
            x_lower, y_lower, x_upper, y_upper = make_box_coords(coord, img_shape=img_shape)
            cur_img[y_lower:y_upper, x_lower:x_upper] = rand_patch
            chamber_stats[idx, ] = [chamber_mean, chamber_std, chamber_shape[0]]

    np.save(cleaned_npy, raw_images)
    np.save(chamber_stats_npy, chamber_stats)
    return raw_images_old, converted_imgs, img_names, img_preds, raw_images, chamber_stats


def get_patch_stats(patch):
    return np.mean(patch), np.std(patch), patch.shape


def find_chamber_center(img_pred):
    pred_threshold = 0.9  # very conservative threshold
    chamber_limits = np.argwhere(img_pred > pred_threshold)  # n*2 where either 1st/2nd coord below pred_threshold

    # mean_y, mean_x = np.mean(chamber_limits, axis=0)
    # better way to find center
    mean_x = np.mean(chamber_limits[:, 1])  # careful about meaning of coordinates: 1st coord is y-axis
    # mean_y = np.percentile(chamber_limits[:, 0], q=25)
    mean_y = np.mean(chamber_limits[chamber_limits[:, 1] == int(mean_x), 0])  # mid-y for central x
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
def get_rand_patch(cur_img, chamber_limits, mean_x, mean_y, patch_size=ACCELL_DIAMETER, centralize_factor=1):
    # random point in thresholded prediction (scaled/downsampled) area
    box_idx = random.randint(0, chamber_limits.shape[0] - 1)  # random point inside mask 512*512
    cur_coord = chamber_limits[box_idx, ]
    cur_y, cur_x = cur_coord

    # move towards image center - if centralize_factor > 0
    cur_x = cur_x + centralize_factor * patch_size / 2. * np.sign(mean_x - cur_x)
    cur_y = cur_y + centralize_factor * patch_size / 2. * np.sign(mean_y - cur_y)

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


# grab and store all accell patches and chamber info
def get_all_ac_cells(seg_folder, img_folder, visualise=False):
    raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats = get_scrubbed_imgs(img_folder)
    cell_size = ACCELL_DIAMETER

    cell_npy = '{}/cell_{}.npy'.format(npy_data_folder, cell_size)
    cell_stats_npy = '{}/cell_stats_{}.npy'.format(npy_data_folder, cell_size)
    if os.path.isfile(cell_npy) and os.path.isfile(cell_stats_npy):
        all_cell_data = np.load(cell_npy)
        all_cell_stats = np.load(cell_stats_npy)
    else:
        all_cell_data = np.ndarray((0, cell_size, cell_size), dtype=np.uint8)
        all_cell_stats = []

        # avg_intensity = np.mean(raw_images)   # adjust for average intensity
        for idx, img_name in enumerate(img_names):  # get accells for segmented image
            sample_name = parse_img_base_name(img_name)
            json_path = '{}/{}.json'.format(seg_folder, sample_name)
            coords = get_coords(json_path)
            cur_image = raw_images[idx, ]   # since accell segmentation on 1024*1000 png
            img_intensity_mean, img_intensity_std, img_shape = get_patch_stats(cur_image)   # img_stats - probably less relevant than chamber_stats

            img_cell_data = np.ndarray((len(coords), cell_size, cell_size), dtype=np.uint8)
            for jdx, coord in enumerate(coords):
                cur_x, cur_y = coord
                # record re-centered cell centers for each file - needed for validation on real data
                cur_cell, cell_coords, old_cell = recenter_cell(cur_image, cur_x, cur_y, img_shape)
                img_cell_data[jdx, ] = cur_cell
                cell_intensity_mean, cell_intensity_std, cell_shape = get_patch_stats(cur_cell)
                cur_cell_stats = (img_intensity_mean, img_intensity_std, cell_intensity_mean, cell_intensity_std,
                                  chamber_stats[idx, 0], chamber_stats[idx, 1])
                all_cell_stats.append(cur_cell_stats)
            all_cell_data = np.append(all_cell_data, img_cell_data, axis=0)

        np.save(cell_npy, all_cell_data)
        all_cell_stats = np.asarray(all_cell_stats, dtype=np.float32)
        np.save(cell_stats_npy, all_cell_stats)

    # some percentile stats for cell data
    print(np.percentile(all_cell_stats, [5, 10, 25, 50, 75, 90, 95], axis=0))
    print(np.percentile(all_cell_stats[:, 2], [5, 10, 25, 50, 75, 90, 95]))
    return all_cell_data, all_cell_stats


def get_max_coord_from_patch(img):
    # cur_coord = img.argmax(axis=0)
    i, j = np.unravel_index(img.argmax(), img.shape)
    return i, j


def get_best_local_cell(img, cur_x, cur_y, num_pixels=2, visualise=False):
    cell_size = ACCELL_DIAMETER
    image_shape = img.shape
    avg_intensities = np.zeros((num_pixels*2+1, num_pixels*2+1), dtype=np.float32)
    search_range = range(-num_pixels, num_pixels+1, 1)
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

    if visualise:
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img.reshape(RAW_IMG_ROWS, RAW_IMG_COLS))
        ax1.scatter(x=cur_x, y=cur_y, c='red', s=2)  # not flipped
        (init_x1, init_y1, init_x2, init_y2) = make_box_coords((cur_x, cur_y), image_shape, box_size=cell_size)
        ax1.add_patch(patches.Rectangle((init_x1, init_y1), cell_size, cell_size, fill=False, color='white'))
        ax1.add_patch(patches.Rectangle((best_cell_coords[0], best_cell_coords[1]), cell_size, cell_size, fill=False, color='red'))

    return best_cell, best_cell_coords, best_x, best_y


def recenter_cell(img, cur_x, cur_y, image_shape, visualise=False):
    cell_size = ACCELL_DIAMETER
    init_cell_coords = make_box_coords((cur_x, cur_y), image_shape, box_size=cell_size)
    init_cell = img[init_cell_coords[1]:init_cell_coords[3], init_cell_coords[0]:init_cell_coords[2]]  # NB. meaning of coords
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
        new_cell, new_cell_coords, new_x, new_y = get_best_local_cell(img, cur_x, cur_y, num_pixels=2)

    if visualise:
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img.reshape(RAW_IMG_ROWS, RAW_IMG_COLS))
        ax1.scatter(x=cur_x, y=cur_y, c='red', s=2)  # not flipped
        (init_x1, init_y1, init_x2, init_y2) = init_cell_coords
        ax1.add_patch(patches.Rectangle((init_x1, init_y1), cell_size, cell_size, fill=False, color='orange'))  # x and y axis make sense here

        ax1.scatter(x=new_x, y=new_y, c='white', s=2)  # not flipped
        (real_x1, real_y1, real_x2, real_y2) = new_cell_coords
        ax1.add_patch(patches.Rectangle((real_x1, real_y1), cell_size, cell_size, fill=False, color='white'))
    return np.copy(new_cell), new_cell_coords, np.copy(init_cell)


# create training data of ac cells on background ac chamber patches
def create_accell_data(patch_rows=32, patch_cols=32, num_samples=10000, visualise=False):
    # cell data with stats of img and chamber where they are from
    all_cell_data, all_cell_stats = get_all_ac_cells(segmentation_json_folder, img_folder, visualise=visualise)
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
                       num_samples=10000, patch_rows=32, patch_cols=32, is_train=True, visualise=False):
    num_imgs = cleaned_imgs.shape[0]

    for idx in range(num_samples):
        # pick random img
        sample_idx = random.randint(0, num_imgs - 1)
        cur_img = cleaned_imgs[sample_idx, ]
        scaled_img = converted_imgs[sample_idx, ]
        img_pred = img_preds[sample_idx, ]
        rand_patch, patch_coords, scaled_coords, chamber_mean, chamber_std, chamber_shape \
            = img_sampler(cur_img, scaled_img, img_pred, patch_size=patch_rows, centralize_factor=1, all_in=True)
        cur_img, coords, mid_coords, cell_types = create_accell_img(rand_patch, chamber_stats[sample_idx, ], cells_data, cells_stats)

        if is_train:
            img_path = os.path.join(output_folder, 'train', 'training_{}.png'.format(idx))
        else:
            img_path = os.path.join(output_folder, 'valid', 'test_{}.png'.format(idx))
        cv2.imwrite(img_path, cur_img)

        if visualise:
            plt.imshow(cur_img)
            plt.scatter(x=mid_coords[:, 0], y=mid_coords[:, 1], c='red', s=1)

        with open(os.path.join(output_folder, '{}_coords.txt'.format('training' if is_train else 'valid')), 'a') as fout:
            for jdx, coord in enumerate(coords):
                # vals = [img_path, str(coord[0]), str(coord[1]), str(coord[2]), str(coord[3]), 'cell']
                vals = [img_path, str(coord[0]), str(coord[1]), str(coord[2]), str(coord[3]), cell_types[jdx]]
                # print(','.join(vals))
                fout.write('{}\n'.format(','.join(vals)))
    return


# plonk cells onto img/patch
def create_accell_img(img, img_chamber_stats, cells_data, cells_stats, visualise=False):
    num_cells, _, _ = cells_data.shape
    num_rows, num_cols = img.shape
    raw_img = np.copy(img)

    # add cells
    cell_size = ACCELL_DIAMETER
    max_cells = num_rows*num_cols / (cell_size * cell_size)
    cell_upper = int(max_cells * .3*.5)     # up to 32*32/25*.3*.5 = 12*.5 = 6 cells per image
    # cell_lower = int(max_cells * .1*.5)    # at least 1 cell in image for useful training
    cell_lower = 3
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
            if img[y_lower:y_upper, x_lower:x_upper].shape != (cell_size, cell_size):
                print('new_x={}; new_y={}; new_coord={}'.format(new_x, new_y, new_coord))
            img, cell_type = set_cell(img, rand_cell, (x_lower, y_lower, x_upper, y_upper), img_chamber_stats, cell_stats,
                           augment=True, blur=False)
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


# careful about flipping patches
def set_cell(img, cell, cell_coords, img_chamber_stats, cell_stats, augment=True, blur=True, visualise=False):
    x_lower, y_lower, x_upper, y_upper = cell_coords
    # img[y_lower:y_upper, x_lower:x_upper] = cell

    # adjust for intensity differences of patch chamber vs cell chamber
    old_cell = np.copy(cell)
    (cell_img_imean, cell_img_istd, cell_imean, cell_istd, cell_chamber_imean, cell_chamber_istd) = cell_stats
    img_chamber_imean, img_chamber_istd, img_chamber_rows = img_chamber_stats
    adjust_factor = float(img_chamber_imean/cell_chamber_imean)

    if augment:
        adjust_factor *= np.random.uniform(low=0.8, high=1.2)
    cell = np.round(cell * adjust_factor)  # adjust cell intensity
    cell = np.clip(cell, 0., 255.)

    cell_type = 'cell_lite'
    if np.mean(cell) > 1.8*np.mean(img):  # these should be clear cells
        cell_type = 'cell'
    elif np.mean(cell) > 1.50*np.mean(img):  # cell 25%=40.96 / chamber 25%=26.88
        cell_type = 'cell_medium'

    # FIXME - adjust for cell std vis-a-vis its chamber vs patch chamber??
    # something like cell2 = np.round((cell - cell_chamber_imean)/cell_chamber_istd * img_chamber_std + img_chamber_imean)
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
        img2 = cv2.blur(img, (2, 2))
        img3 = np.copy(img)
        img[y_lower-1:y_lower+1, x_lower:x_upper] = img2[y_lower-1:y_lower, x_lower:x_upper]
        img[y_upper-1:y_upper+1, x_lower:x_upper] = img2[y_upper-1:y_upper+1, x_lower:x_upper]
        img[y_lower:y_upper, x_lower-1:x_lower+1] = img2[y_lower:y_upper, x_lower-1:x_lower+1]
        img[y_lower:y_upper, x_upper-1:x_upper+1] = img2[y_lower:y_upper, x_upper-1:x_upper+1]

        if visualise:
            plt.figure(1); plt.clf()
            plt.imshow(img3)
            plt.scatter(x=[x_lower, x_upper], y=[y_lower, y_upper], c='red', s=3)
            plt.figure(2); plt.clf()
            plt.imshow(img2)
            plt.scatter(x=[x_lower, x_upper], y=[y_lower, y_upper], c='red', s=3)
            plt.figure(3); plt.clf()
            plt.imshow(img)
            plt.scatter(x=[x_lower, x_upper], y=[y_lower, y_upper], c='red', s=3)
            smoothed_coords = np.argwhere(img!=img3)
            plt.scatter(x=smoothed_coords[:, 1], y=smoothed_coords[:, 0], c='yellow', s=1)
    return img, cell_type


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


def get_larger_chambers(img_folder, accell_json_folder=segmentation_json_folder):
    raw_images, converted_imgs, img_names, img_preds, cleaned_imgs, chamber_stats \
        = get_scrubbed_imgs(img_folder, accell_json_folder=segmentation_json_folder)

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


if __name__ == '__main__':
    # # predict all inflamed ac chambers
    # raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder=img_folder)
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
    all_cell_data, all_cell_stats \
        = get_all_ac_cells(seg_folder=segmentation_json_folder, img_folder=img_folder, visualise=False)
    #
    # # patch_sampler - check for erroneous/edge patches and too high intensity
    # # plunk cells in patch from patch_sampler - adjust for relative intensity of original chamber vs patch chamber
    # create_accell_imgs(cleaned_imgs, converted_imgs, img_preds, chamber_stats, all_cell_data, all_cell_stats,
    #                    output_folder='./accell/new_32_32/',
    #                    num_samples=10000, patch_rows=32, patch_cols=32, is_train=True, visualise=False)

    # create training and validation data
    create_accell_data(patch_rows=32, patch_cols=32, num_samples=20000, visualise=False)