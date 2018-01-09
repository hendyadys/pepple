import sys, os, glob, cv2, json
import numpy as np
import random

from matplotlib import pyplot as plt
from matplotlib import patches
# plt.switch_backend('agg')
from scipy import ndimage

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
    # linux
    segmentation_json_folder = '/home/ayl/data/pepple/accell/jsons'
    img_folder = '/home/ayl/data/pepple/accell/segmentations'
    empty_img_folder = '/home/ayl/data/pepple/accell/empty_segmentations'

    output_folder = '/home/yue/pepple/accell/processed'
    output_folder_test = '/home/yue/pepple/accell/test'
    output_folder2 = '/home/yue/pepple/accell/processed2'   # to conform to augmented data dimensions
    output_folder_test2 = '/home/yue/pepple/accell/test2'   # to conform to augmented data dimensions
    augmented_folder = '/home/yue/pepple/accell/augmented'
    augmented_folder_test = '/home/yue/pepple/accell/augmented_test'
    augmented_folder_master = '/home/yue/pepple/accell/augmented_master'

    chamber_weights_folder = '/home/yue/pepple/runs/2017-08-09-10-20-24/weights'
elif platform == "win32":
    # Windows...
    segmentation_json_folder = './accell/jsons'
    img_folder = './accell/segmentations'
    empty_img_folder = './accell/empty_segmentations'

    output_folder = './accell/processed'
    output_folder_test = './accell/test'
    output_folder2 = './acell/processed2'   # to conform to augmented data dimensions
    output_folder_test2 = './accell/test2'   # to conform to augmented data dimensions
    augmented_folder = './accell/augmented'
    augmented_folder_test = './accell/augmented_test'
    augmented_folder_master = './accell/augmented_master'

    chamber_weights_folder = './runs/runVertical/weights'

test_weights = 'weights-improvement-050--0.95407502.hdf5'
# test_weights = 'weights-improvement-099--0.95389259.hdf5'
# test_weights = 'weights-improvement-088--0.95875288.hdf5'

RAW_IMAGE_HEIGHT = 1024
RAW_IMAGE_WIDTH = 1000
DOWNSAMPLE_RATIO = 2.
DS_IMAGE_HEIGHT = int(RAW_IMAGE_HEIGHT/DOWNSAMPLE_RATIO)
DS_IMAGE_WIDTH = int(RAW_IMAGE_WIDTH/DOWNSAMPLE_RATIO)
IMAGE_WIDTH = 512   # predicted (for chamber segmentation)
IMAGE_HEIGHT = 512
nrows=496*2   # correspond to y-axis
ncols=128   # correspond to x-axis
step_size_r=8*2
step_size_c=8*2
cell_size = 5  # make it look like islands
NOISIER = True

# for sampler
SAMPLER_CLEANED_IMGS = None
SAMPLER_PREDS = None
SAMPLER_IMG_NAMES = []


def visualize_bbox_data(img, img_coords, cur_img, cur_coords, cur_x, cur_y, nrows=nrows, ncols=ncols, flipAxis=False):
    # view large image - with coords
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    # show img strip
    ax1.add_patch(patches.Rectangle((cur_x, cur_y), ncols, nrows, fill=False, color='red'))

    for coord in img_coords:    # add cells to image
        (xs, ys) = coord['mousex'], coord['mousey']
        if flipAxis:    # shouldnt need to flip
            ax1.scatter(x=ys, y=xs, c='red', s=2)  # be careful of orientation - imshow flips coords
        else:
            ax1.scatter(x=xs, y=ys, c='red', s=2)  # not flipped

    # np.sum(img[cur_y:cur_y+nrows, cur_x:cur_x+ncols] == cur_img)
    # view cropped image in situ - with coords
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(cur_img)
    for coord in cur_coords:    # add cells to image
        (real_x1, real_y1, real_x2, real_y2) = coord    # these bounding box processed
        if flipAxis:    # imshow flips coords - shouldnt need to flip
            ax1.add_patch(
                patches.Rectangle((real_y1, real_x1), real_y2 - real_y1, real_x2 - real_x1, fill=False, color='green'))
        else:
            ax1.add_patch(
                patches.Rectangle((real_x1, real_y1), real_x2 - real_x1, real_y2 - real_y1, fill=False, color='green'))
            ax1.scatter(x=[(real_x1+real_x2)/2], y=[(real_y1+real_y2)/2], c='red', s=2)
    return


def make_ac_coords(img_coords, cur_x, cur_y, num_cols=ncols, num_rows=nrows):
    cur_coords = []
    for coord in img_coords:
        xs = coord['mousex']
        ys = coord['mousey']
        for idx, x in enumerate(xs):
            y = ys[idx]
            if (x>=cur_x and x<cur_x+num_cols) and (y>=cur_y and y<cur_y+num_rows):  # if within bounds then process
                1
            else:
                if (x >= cur_x and x < cur_x + ncols) and (y >= cur_y and y < cur_y + nrows):   # incorrect old code
                    2
                continue

            lower_x = max(x-(cell_size//2), cur_x) - cur_x
            upper_x = min(x+(cell_size//2), cur_x+ncols) - cur_x

            lower_y = max(y-(cell_size//2), cur_y) - cur_y
            upper_y = min(y+(cell_size//2), cur_y+nrows) - cur_y
            cur_coords.append((lower_x, lower_y, upper_x, upper_y))     # imshow flips coords
    return cur_coords


def create_cropped_images(base_name, img, img_coords, num_rows=nrows, num_cols=ncols, is_augmented=False, do_write=True, visualise=False):
    img_shape = img.shape
    for y in range(0, img_shape[0]-num_rows+1, step_size_r):   # y-axis
        for x in range(0, img_shape[1]-num_cols+1, step_size_c):   # x-axis
            cur_img = img[y:y+num_rows, x:x+num_cols]     # height(y) by width(x)
            # shouldn't hit this bit
            cur_shape = cur_img.shape
            if not (cur_shape[0]==num_rows and cur_shape[1]==num_cols):   # undersized strip and ignore
                continue

            if do_write:
                if is_augmented==0:
                    cur_name = '{}/{}_{}_{}.png'.format(output_folder, base_name, x, y)
                    outfile = './{}.txt'.format('training_coords')
                elif is_augmented==2:   # for original labeled ac cell images
                    cur_name = '{}/{}_{}_{}.png'.format(output_folder2, base_name, x, y)
                    outfile = '{}/{}.txt'.format(output_folder2, 'training_coords')
                else:   # for augmented/generated ac cell images
                    cur_name = '{}/{}_{}_{}.png'.format(augmented_folder, base_name, x, y)
                    outfile = '{}/{}.txt'.format(augmented_folder_master,'training_coords')
            else:
                if is_augmented==0:
                    cur_name = '{}/{}_{}_{}.png'.format(output_folder_test, base_name, x, y)
                    outfile = './{}.txt'.format('test_coords')
                elif is_augmented==2:
                    cur_name = '{}/{}_{}_{}.png'.format(output_folder_test2, base_name, x, y)
                    outfile = '{}/{}.txt'.format(output_folder2, 'test_coords')
                else:
                    cur_name = '{}/{}_{}_{}.png'.format(augmented_folder_test, base_name, x, y)
                    outfile = '{}/{}.txt'.format(augmented_folder_master, 'test_coords')

            cv2.imwrite(cur_name, cur_img)
            with open(outfile, 'a') as fout:
                cur_coords = make_ac_coords(img_coords, x, y, num_cols=num_cols, num_rows=num_rows)
                # np.max(cur_coords, axis=0)
                # np.max(np.asarray(cur_coords), axis=0)
                if len(cur_coords) > 0 and np.max(cur_coords) > 320:    # shouldn't happen
                    print('error in making cell coords for {}'.format(cur_name))
                for coord in cur_coords:
                    vals = [cur_name, str(coord[0]), str(coord[1]), str(coord[2]), str(coord[3]), 'cell']
                    print(','.join(vals))
                    fout.write('{}\n'.format(','.join(vals)))

            if do_write and visualise and len(cur_coords)>0:
                visualize_bbox_data(img, img_coords, cur_img, cur_coords, x, y, nrows=num_rows, ncols=num_cols)
    return


# for each seg and corresponding image
# make strip data
# and record labeled bounding boxes in text file
# keep valid/training images distinct
def make_test_images(is_aug=0, nrows=nrows, ncols=ncols):
    pnames = sorted(list(glob.glob('{}/*.json'.format(segmentation_json_folder))))

    seen_names = []
    num_samples = 0
    sample_threshold = len(pnames)*.8
    for pname in pnames:
        sample_name = pname.split('mouse')[1]
        if sample_name in seen_names: continue  # no duplicate samples
        else:
            seen_names.append(sample_name)
            num_samples +=1

        coord_data = get_coords(pname)

        # corresponding image
        base_name = pname.split('.json')[0].replace('{}/'.format(segmentation_json_folder), '')
        # base_name = pname.split('.json')[0].replace('{}\\'.format(segmentation_json_folder), '')
        img_path = make_img_name(base_name)
        # print(base_name, img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # create cropped images and write to text file
        print(pname, num_samples, sample_threshold, num_samples<sample_threshold)
        create_cropped_images(base_name, img, coord_data, num_rows=nrows, num_cols=ncols, is_augmented=is_aug, do_write=num_samples<sample_threshold)
    return 1


def make_img_name(base_name):
    # img_path = base_name.replace(segmentation_json_folder, img_folder)
    # image_path = '{}.png'.format(img_path)
    image_path = '{}/{}.png'.format(img_folder, base_name)
    return image_path


def get_coords(coord_file):
    fin = open(coord_file).read()
    json_data = json.loads(fin)
    return json_data


def check_json_consistency():
    pnames = sorted(list(glob.glob('{}/*.json'.format(segmentation_json_folder))))
    total_coords = 0
    for pname in pnames:
        coord_data = get_coords(pname)
        for coord in coord_data:
            total_coords +=len(coord['mousex'])
            if len(coord['mousex'])>1 or len(coord['mousey'])>1:
                print(pname, coord['mousex'], coord['mousey'])
    return total_coords     # about 2531 cells over 216(1024*1000) files (0.2%)


def add_sim_ac_cells(img, img_coords, num_new):
    # avoid overlapping points
    # avoid high intensity areas
    # choose coords inside chamber or just randomly?
    # cell intensities
    return


## augment data
# get img data for both segmented and empty (no ac cell) chambers
def get_accell_img_data(mode='empty'):
    if mode=='empty':
        base_img_folder = empty_img_folder
        ds_filename = '{}/downsampled_images_empty_seg.npy'.format(augmented_folder_master)
        raw_filename = '{}/raw_images_empty_seg.npy'.format(augmented_folder_master)
    else:
        base_img_folder = img_folder
        ds_filename = '{}/downsampled_images_seg.npy'.format(augmented_folder_master)
        raw_filename = '{}/raw_images_seg.npy'.format(augmented_folder_master)

    pnames = sorted(list(glob.glob('{}/*.png'.format(base_img_folder))))
    num_images = int(len(pnames)/2)
    if os.path.isfile(ds_filename) and os.path.isfile(raw_filename):    # just load
        ds_imgs = np.load(ds_filename)
        raw_imgs = np.load(raw_filename)
        real_pnames = [x for x in pnames if 'mask' not in x]
    else:
        raw_imgs = np.ndarray((num_images, RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH, 1), dtype=np.uint8)
        count = 0
        real_pnames = []
        for idx, pname in enumerate(pnames):
            if 'mask' in pname: continue

            real_pnames.append(pname)
            img = cv2.imread(pname, cv2.IMREAD_GRAYSCALE)
            raw_imgs[count, :, :, 0] = img
            count += 1
        ds_imgs = downsample(raw_imgs, factor=int(DOWNSAMPLE_RATIO))

        np.save(raw_filename, raw_imgs)
        np.save(ds_filename, ds_imgs)
    return real_pnames, ds_imgs, raw_imgs


# naive downsampling; maybe better to average pixels
def downsample(imgs, factor=int(DOWNSAMPLE_RATIO)):
    downsampled_imgs = imgs[:, ::factor, ::factor]
    return downsampled_imgs


def parse_img_base_name(pname):
    if platform == "linux" or platform == "linux2":
        p_toks = pname.split('/')
    elif platform == "win32":
        p_toks = pname.split('\\')
    sample_name = p_toks[-1].replace('.png', '')
    return sample_name


def get_all_ac_cells(pnames, raw_data, threshold=50, visualise=False):
    if threshold:
        cell_outfile = '{}/coords_neighbours_{}_t{}.npy'.format(augmented_folder_master, cell_size, threshold)
    else:
        cell_outfile = '{}/coords_neighbours_{}.npy'.format(augmented_folder_master, cell_size)
    if os.path.isfile(cell_outfile):
        all_coords_data = np.load(cell_outfile)
    else:
        all_coords_data = np.ndarray((0, cell_size, cell_size), dtype=np.uint8)
        avg_intensity = np.mean(raw_data)   # adjust for average intensity
        count = 0
        for pname in pnames:
            if 'mask' in pname: continue

            sample_name = parse_img_base_name(pname)
            json_path = '{}/{}.json'.format(segmentation_json_folder, sample_name)
            coord_data = get_coords(json_path)
            cur_image = raw_data[count,]
            image_shape = cur_image.shape
            img_intensity = np.mean(cur_image)
            count +=1

            img_coord_data = np.ndarray((0, cell_size, cell_size), dtype=np.uint8)
            for coord in coord_data:
                xs, ys = coord['mousex'], coord['mousey']
                for j, cur_x in enumerate(xs):
                    cur_y = ys[j]
                    # FIXME - record re-centered cell centers for each file - only needed for validation on real data
                    cur_cell, cell_coords, old_cell = recenter_cell(cur_image, cur_x, cur_y, image_shape)
                    cur_cell = np.round(cur_cell/img_intensity*avg_intensity)   # adjust for img vs avg intensity
                    if threshold and np.mean(cur_cell) > threshold:
                        img_coord_data = np.append(img_coord_data, cur_cell.reshape(1, cell_size, cell_size), axis=0)
                    else:
                        img_coord_data = np.append(img_coord_data, cur_cell.reshape(1, cell_size, cell_size), axis=0)
                    # np.mean(cur_cell) # should be much higher than np.mean(cur_image)

            all_coords_data = np.append(all_coords_data, img_coord_data, axis=0)
            # # debug on intensities
            # np.mean(all_coords_data)
            # np.mean(all_coords_data, axis=(1, 2))
            # np.mean(raw_data)
            # np.percentile(all_coords_data.reshape(len(all_coords_data), 25), [10, 50, 90], axis=1)
            # np.percentile(all_coords_data, [10, 50, 90], axis=(1, 2))

        np.save(cell_outfile, all_coords_data)
    return all_coords_data


def get_best_local_cell(img, cur_x, cur_y, num_pixels=2, visualise=False):
    image_shape = img.shape
    avg_intensities = np.zeros((num_pixels*2+1, num_pixels*2+1), dtype=np.float32)
    search_range = range(-num_pixels, num_pixels+1, 1)
    for idx, dx in enumerate(search_range):
        for jdy, dy in enumerate(search_range):
            new_x = cur_x + dx
            new_y = cur_y + dy
            new_cell_coords = make_box_coords(new_x, new_y, image_shape, box_size=cell_size)
            new_cell = img[new_cell_coords[1]:new_cell_coords[3], new_cell_coords[0]:new_cell_coords[2], 0]
            avg_intensities[idx, jdy] = np.mean(new_cell)

    i, j = np.unravel_index(avg_intensities.argmax(), avg_intensities.shape)
    best_x = cur_x +search_range[i]
    best_y = cur_y + search_range[j]
    best_cell_coords = make_box_coords(best_x, best_y, image_shape, box_size=cell_size)
    best_cell = img[best_cell_coords[1]:best_cell_coords[3], best_cell_coords[0]:best_cell_coords[2], 0]

    if visualise:
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img.reshape(RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH))
        ax1.scatter(x=cur_x, y=cur_y, c='red', s=2)  # not flipped
        (init_x1, init_y1, init_x2, init_y2) = make_box_coords(cur_x, cur_y, image_shape, box_size=cell_size)
        ax1.add_patch(patches.Rectangle((init_x1, init_y1), cell_size, cell_size, fill=False, color='white'))
        ax1.add_patch(patches.Rectangle((best_cell_coords[0], best_cell_coords[1]), cell_size, cell_size, fill=False, color='red'))

    return best_cell, best_cell_coords, best_x, best_y


def recenter_cell(img, cur_x, cur_y, image_shape, visualise=False):
    init_cell_coords = make_box_coords(cur_x, cur_y, image_shape, box_size=cell_size)
    init_cell = img[init_cell_coords[1]:init_cell_coords[3], init_cell_coords[0]:init_cell_coords[2], 0]  # NB. meaning of coords
    # cur_cell = img[cur_cell_coords[0]:cur_cell_coords[2], cur_cell_coords[1]:cur_cell_coords[3], 0]

    # get maximal intensity location and re-center
    old_search = False
    if old_search:
        cell_y, cell_x = get_max_coord_from_patch(init_cell)
        new_y = cur_y + cell_y - cell_size//2
        new_x = cur_x + cell_x - cell_size//2

        new_cell_coords = make_box_coords(new_x, new_y, image_shape, box_size=cell_size)
        new_cell = img[new_cell_coords[1]:new_cell_coords[3], new_cell_coords[0]:new_cell_coords[2], 0]  # NB. meaning of coords
        if np.mean(new_cell) < np.mean(init_cell):  # want best intensity patches
            new_cell = init_cell
            new_cell_coords = init_cell_coords
    else:
        new_cell, new_cell_coords, new_x, new_y = get_best_local_cell(img, cur_x, cur_y, num_pixels=2)

    if visualise:
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img.reshape(RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH))
        ax1.scatter(x=cur_x, y=cur_y, c='red', s=2)  # not flipped
        (init_x1, init_y1, init_x2, init_y2) = init_cell_coords
        ax1.add_patch(patches.Rectangle((init_x1, init_y1), cell_size, cell_size, fill=False, color='orange'))  # x and y axis make sense here

        ax1.scatter(x=new_x, y=new_y, c='white', s=2)  # not flipped
        (real_x1, real_y1, real_x2, real_y2) = new_cell_coords
        ax1.add_patch(patches.Rectangle((real_x1, real_y1), cell_size, cell_size, fill=False, color='white'))
    return np.copy(new_cell), new_cell_coords, np.copy(init_cell)


def get_max_coord_from_patch(img):
    # cur_coord = img.argmax(axis=0)
    i,j = np.unravel_index(img.argmax(), img.shape)
    return i,j


def make_box_coords(cur_x, cur_y, img_shape, box_size=cell_size):
    img_height = img_shape[0]
    img_width = img_shape[1]

    lower_x = max(cur_x - (box_size // 2), 0)
    upper_x = min(cur_x + (box_size // 2)+1, img_width)

    lower_y = max(cur_y - (box_size // 2), 0)
    upper_y = min(cur_y + (box_size // 2)+1, img_height)
    return lower_x, lower_y, upper_x, upper_y


def make_augmented_image_data(visualise=False):
    # # get empty segmentation images as clean canvas for adding cells - TOO many collapsed chambers
    # empty_img_names, ds_imgs_empty, raw_imgs_empty = get_accell_img_data(mode='empty')
    # get segmented ac cell images for cell info
    seg_img_names, ds_imgs_seg, raw_imgs_seg = get_accell_img_data(mode='segmented')
    # remove cells from ac cell segmented as these imgs have best chambers
    raw_imgs_cleaned = remove_cells(raw_imgs_seg, seg_img_names)
    ds_imgs_empty = downsample(raw_imgs_cleaned, factor=int(DOWNSAMPLE_RATIO))
    empty_img_names = seg_img_names
    ds_filename = '{}/downsampled_images_clean.npy'.format(augmented_folder_master)
    raw_filename = '{}/raw_images_clean.npy'.format(augmented_folder_master)
    np.save(raw_filename, raw_imgs_cleaned)
    np.save(ds_filename, ds_imgs_empty)

    # grab all cell data for random sampling and adding to images
    all_cell_data = get_all_ac_cells(seg_img_names, raw_imgs_seg, threshold=0, visualise=False)
    # all_cell_data = get_all_ac_cells(seg_img_names, raw_imgs_seg, threshold=75, visualise=False)

    # predict chamber for empty images
    ds_combined_preds = combine_predicted_chambers(ds_imgs_empty)
    ds_combined_preds = ds_combined_preds[:, :, :DS_IMAGE_WIDTH]   # can remove the padding for preds

    # divide into training vs validation
    train_valid_split = .8
    num_imgs_train = int(len(empty_img_names)*train_valid_split)
    img_names_train = empty_img_names[:num_imgs_train]
    ds_imgs_train = ds_imgs_empty[:num_imgs_train, ]
    # for chamber coords and seeding cells properly
    ds_preds_train = ds_combined_preds[:num_imgs_train, ]
    num_coords_train = int(len(all_cell_data)*train_valid_split)
    cell_train = all_cell_data[:num_coords_train, ]
    make_augmented_data_helper(img_names_train, ds_imgs_train, ds_preds_train, cell_train, mode='training')

    img_names_valid = empty_img_names[num_imgs_train:]
    ds_imgs_valid = ds_imgs_empty[num_imgs_train:, ]
    ds_preds_valid = ds_combined_preds[num_imgs_train:, ]
    cell_valid = all_cell_data[num_coords_train:, ]
    make_augmented_data_helper(img_names_valid, ds_imgs_valid, ds_preds_valid, cell_valid, mode='validation')
    return


def make_augmented_data_helper(img_names, ds_imgs, ds_combined_preds, cell_data, mode='training'):
    write_option = True if mode=='training' else False
    count = 0
    for idx, pname in enumerate(img_names):
        if 'mask' in pname: continue

        pred_mask = ds_combined_preds[count, ]
        raw_img = ds_imgs[count, :, :, 0]
        augmented_img, coords = augment_img(raw_img, pred_mask, cell_data)

        # save image and coords
        base_name = parse_img_base_name(pname)
        img_name = '{}/{}_master.png'.format(augmented_folder_master, base_name)
        cv2.imwrite(img_name, augmented_img)

        # create strips from augmented images
        coords_old_format = old_format(coords)
        create_cropped_images(base_name, augmented_img, coords_old_format, num_rows=320, num_cols=128, is_augmented=True,
                              do_write=write_option)
        count +=1

    return


def remove_cells(raw_imgs, img_names):
    out_imgs = np.copy(raw_imgs)
    for idx, img in enumerate(out_imgs):
        pname = img_names[idx]
        base_name = parse_img_base_name(pname)
        json_file = '{}/{}.json'.format(segmentation_json_folder, base_name)
        coord_data = get_coords(json_file)

        img_mean = np.mean(img)
        img_std = np.std(img)
        for coord in coord_data:
            xs = coord['mousex']
            ys = coord['mousey']
            for jdx, x in enumerate(xs):
                y = ys[jdx]
                img[y-4:y+4, x-4:x+4, 0] = np.random.normal(img_mean, img_std/3, (8, 8))    # wipe out big area
    return out_imgs


def old_format(coords):
    old_coords = []
    for coord in coords:
        cur_coord = {'mousex':[coord[1]], 'mousey':[coord[0]], 'mousetime':[]}  # flip back coords after chamber_limits
        old_coords.append(cur_coord)
    return old_coords


def predict_chamber(img_data, all_preds_path, chamber_weights_folder=chamber_weights_folder, test_weights=test_weights):  # need to be downsampled to 512*500
    from train import get_unet
    from data import slice_data
    from analyser import center_scale_imgs

    scaled_img, _, params = center_scale_imgs(img_data, img_data)
    sliced_img, sliced_mask = slice_data(scaled_img, scaled_img)  # since predicting don't care about mask
    # sliced_img, sliced_mask = slice_data(img_data, img_data)    # since predicting don't care about mask
    model = get_unet()
    weight_file = '%s/%s' % (chamber_weights_folder, test_weights)  # use 50th epoch for sanity
    model.load_weights(weight_file)
    output = model.predict(sliced_img, batch_size=10)  # inference step
    np.save(all_preds_path, output)
    return output


def combine_predicted_chambers(ds_imgs, all_preds_file='all_preds.npy', combined_preds_file='combined_preds.npy'):
    from analyser import combine_img, num_aug, img_rows, img_cols

    all_preds_path = '{}/{}'.format(augmented_folder_master, all_preds_file)
    combined_preds_path = '{}/{}'.format(augmented_folder_master, combined_preds_file)
    if os.path.isfile(all_preds_path):
        strip_preds = np.load(all_preds_path)
    else:
        strip_preds = predict_chamber(ds_imgs, all_preds_path)

    if os.path.isfile(combined_preds_path):
        all_combined_preds = np.load(combined_preds_path)
    else:
        predicted_shape = strip_preds.shape
        num_images = int(predicted_shape[0]/num_aug)
        all_combined_preds = np.ndarray((num_images, img_rows, img_cols), dtype=np.float32)

        for idx in range(0, predicted_shape[0], num_aug):
            cur_strip_preds = strip_preds[idx:idx+num_aug,]
            cur_combined_pred = combine_img(cur_strip_preds, real_mean=True)    # this pads bottom
            cur_count = int(idx/num_aug)
            all_combined_preds[cur_count, ] = cur_combined_pred

        np.save(combined_preds_path, all_combined_preds)
    return all_combined_preds


def augment_img(raw_img, pred_mask, all_coords_data, visualise=False):
    # for downsampled image randomly add cells
    # get chamber size
    chamber_size, chamber_limits, max_chamber_height = calc_img_chamber_size(pred_mask, remove_edge_coords=True,
                                                                             visualise=True)
    coords = get_augment_coords(pred_mask, chamber_limits, visualise=visualise)     # calculate conservative chamber_size based on chamber_limits
    # add cell patches img as augmentation
    aug_img, used_coords = make_aug_img(raw_img, pred_mask, coords, all_coords_data, visualise=visualise)
    return aug_img, used_coords


def calc_img_chamber_size(mask, remove_edge_coords=True, visualise=False):
    # chamber_size = np.count_nonzero(img)
    # chamber_limits = np.transpose(np.nonzero(img))
    # pred_threshold = 0.3
    pred_threshold = 0.9    # very conservative threshold
    chamber_size = np.sum(mask > pred_threshold)
    chamber_limits = np.argwhere(mask > pred_threshold)    # n*2 where either 1st/2nd coord below pred_threshold

    # remove edge points - assume image centered for training
    if remove_edge_coords:
        edge_limit = 50
        edge_coords_1 = np.argwhere(chamber_limits < edge_limit)    # k*2, where either 0,1
        edge_coords_2 = np.argwhere(chamber_limits > DS_IMAGE_WIDTH - edge_limit)
        edge_coords = np.append(edge_coords_1, edge_coords_2, axis=0)
        chamber_limits = np.delete(chamber_limits, edge_coords[:, 0], axis=0)
        # chamber_limits[edge_coords]   # this doesn't work - need np.where to check actual values
        # edge_coords_y = chamber_limits[np.where(chamber_limits[:,0] < edge_limit),:]
        # edge_coords_x = chamber_limits[np.where(chamber_limits[:,1] < edge_limit),:]

    # find max chamber height
    (min_y, min_x) = np.min(chamber_limits, axis=0)     # think about what y and x-axis mean for image
    (max_y, max_x) = np.max(chamber_limits, axis=0)
    max_chamber_height = max_y-min_y    # care about vertical height more

    if visualise:
        plt.clf()
        plt.imshow(mask)
        plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='red', s=1)  # be careful of orientation - imshow flips coords
    # chamber_limits are flipped vs imshow
    return chamber_size, chamber_limits, max_chamber_height


# TODO - something i dont understand about setting numpy arrays
def get_augment_coords(mask, chamber_limits, visualise=False):
    chamber_size, _ = chamber_limits.shape
    # determine number of cells based on chamber size
    max_cells = chamber_size / (cell_size * cell_size)
    cell_upper = int(max_cells * .3)
    cell_lower = int(max_cells * .1)
    num_samples = random.randint(cell_lower, cell_upper)

    # randomly pick chamber coords
    coord_inds = np.random.choice(chamber_size, num_samples, replace=False)
    # coord_inds = random.sample(range(chamber_size), num_samples)  # equivalent sampling without replacement
    # coords = np.zeros((num_samples, 2), dtype=np.uint8)   # pre-allocate
    # real_count = 0
    # coord_list = []
    coords = np.zeros((0, 2), dtype=np.uint8)  # pre-allocate
    mean_y, mean_x = np.mean(chamber_limits, axis=0)    # careful about meaning of coordinates: 1st coord is y-axis
    for idx, ind in enumerate(coord_inds):
        cur_coord = chamber_limits[ind, ]
        # avoid chamber edge - move towards center
        cur_y, cur_x = cur_coord
        cur_x = int(cur_x + 2*cell_size * np.sign(mean_x-cur_x))
        cur_y = int(cur_y + 2*cell_size * np.sign(mean_y-cur_y))
        new_coord = cur_y, cur_x

        if visualise:
            # plt.clf()
            plt.imshow(mask)
            plt.scatter(x=mean_x, y=mean_y, c='red', s=2)
            # plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='magenta', s=2)
            plt.scatter(x=cur_coord[1], y=cur_coord[0], c='blue', s=2)     # careful about meaning of coordinates
            plt.scatter(x=new_coord[1], y=new_coord[0], c='green', s=2)

        # avoid overlapping
        # overlapping_new = is_overlapping(coords[0:real_count,], new_coord)
        overlapping_new = is_overlapping(coords, new_coord)
        if not overlapping_new:
            # coord_list.append(new_coord)
            # coords[real_count, ] = np.copy(new_coord)
            # coords[real_count, 0] = new_coord[0]
            # coords[real_count, 1] = new_coord[1]
            # real_count +=1
            coords = np.append(coords, np.reshape(np.asarray(new_coord), (1,2)), axis=0)

    if visualise:
        plt.imshow(mask)
        plt.scatter(x=mean_x, y=mean_y, c='red', s=2)
        # plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='cyan', s=1)
        # plt.scatter(x=coords[0:real_count, 1], y=coords[0:real_count, 0], c='magenta', s=2)
        plt.scatter(x=coords[:, 1], y=coords[:, 0], c='magenta', s=2)
    # return coords[0:real_count,]
    return coords   # coords still flipped here


def is_overlapping(coords, new_coord):
    if len(coords):
        dist = np.sum((coords-new_coord)**2, axis=1)
        return np.any(dist<2*(cell_size**2))
    else:
        return False


# careful about flipping patches
def set_cell(img, cell, cell_coords, blur=True, visualise=False):
    x_lower, y_lower, x_upper, y_upper = cell_coords
    img[y_lower:y_upper, x_lower:x_upper] = cell

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
        plt.scatter(x=smoothed_coords[:,1], y=smoothed_coords[:,0], c='yellow', s=1)
    return img


def set_patch(img, patch, patch_coords):
    x_lower, y_lower, x_upper, y_upper = patch_coords
    patch_height, patch_width = patch.shape
    for i in range(patch_width):
        for j in range(patch_height):
            img[y_lower+j, x_lower+i] = patch[j, i]
    return img


def make_aug_img(raw_img, pred_mask, coords, all_coords_data, visualise=False):
    num_patches, _, _ = all_coords_data.shape
    img_shape = raw_img.shape
    aug_img = np.copy(raw_img)
    used_coords = []
    for coord in coords:
        rand_ind = random.choice(range(0, num_patches))
        rand_patch = all_coords_data[rand_ind, ]
        cur_y, cur_x = coord    # height then width since chamber_limits is flipped
        x_lower, y_lower, x_upper, y_upper = make_box_coords(cur_x, cur_y, img_shape=img_shape)
        temp = aug_img[y_lower:y_upper, x_lower:x_upper]
        if temp.shape==rand_patch.shape:
            # check patch set properly
            # raw_img[y_lower:y_upper, x_lower:x_upper]
            # aug_img[y_lower:y_upper, x_lower:x_upper] = rand_patch
            aug_img = set_patch(aug_img, rand_patch, (x_lower, y_lower, x_upper, y_upper))
            # used_coords.append((x_lower, y_lower, x_upper, y_upper))
            used_coords.append((cur_y, cur_x))  # maintain flipped order for now
        else:
            1

        if visualise:
            plt.figure(1)
            plt.imshow(raw_img)
            plt.scatter(x=cur_x, y=cur_y, c='red', s=2)  # coord center
            plt.axes().add_patch(patches.Rectangle((x_lower, y_lower), cell_size, cell_size, fill=False, color='green'))   # show box

            plt.figure(2)
            plt.imshow(pred_mask)
            plt.scatter(x=cur_x, y=cur_y, c='red', s=2)  # coord center
            plt.axes().add_patch(patches.Rectangle((x_lower, y_lower), cell_size, cell_size, fill=False, color='green'))   # show box

            plt.figure(3)
            plt.imshow(aug_img)
            plt.scatter(x=cur_x, y=cur_y, c='red', s=2)  # coord center
            plt.axes().add_patch(patches.Rectangle((x_lower, y_lower), cell_size, cell_size, fill=False, color='green'))   # show box

            # sanity check - highlight bright spots in img
            plt.figure(4)
            plt.imshow(aug_img)
            bright_spots = np.argwhere(aug_img > 75)
            plt.scatter(x=bright_spots[:,1], y=bright_spots[:,0], c='red', s=1)  

    return aug_img, np.asarray(used_coords)


# debug chamber segmentations
def debug_chamber_segmentations():
    seg_img_names, ds_imgs_seg, raw_imgs_seg = get_accell_img_data(mode='segmented')
    cpreds_50 = combine_predicted_chambers(ds_imgs_seg, all_preds_file='all_preds.npy', combined_preds_file='combined_preds.npy')
    cpreds_88 = combine_predicted_chambers(ds_imgs_seg, all_preds_file='all_preds_e88.npy', combined_preds_file='combined_preds_e88.npy')
    cpreds_99 = combine_predicted_chambers(ds_imgs_seg, all_preds_file='all_preds_e99.npy', combined_preds_file='combined_preds_e99.npy')

    cpreds_20 = combine_predicted_chambers(ds_imgs_seg, all_preds_file='all_preds_20.npy', combined_preds_file='combined_preds_e20.npy')
    cpreds_27 = combine_predicted_chambers(ds_imgs_seg, all_preds_file='all_preds_27.npy',
                                           combined_preds_file='combined_preds_27.npy')
    cpreds_62 = combine_predicted_chambers(ds_imgs_seg, all_preds_file='all_preds_62.npy',
                                           combined_preds_file='combined_preds_e62.npy')
    cpreds_378 = combine_predicted_chambers(ds_imgs_seg, all_preds_file='all_preds_378_slow.npy',
                                           combined_preds_file='combined_preds_e378_slow.npy')

    # compare what happens in chamber and save images
    num_files = len(seg_img_names)
    for idx, pname in enumerate(seg_img_names):
        ds_img = ds_imgs_seg[idx, :, :, 0]
        pred_mask_50 = cpreds_50[idx, ]
        pred_mask_88 = cpreds_88[idx, ]
        pred_mask_99 = cpreds_99[idx, ]
        chamber_size_50, chamber_limits_50, max_chamber_height_50 = calc_img_chamber_size(pred_mask_50)
        chamber_size_88, chamber_limits_88, max_chamber_height_88 = calc_img_chamber_size(pred_mask_88)
        chamber_size_99, chamber_limits_99, max_chamber_height_99 = calc_img_chamber_size(pred_mask_99)

        pred_mask_20 = cpreds_20[idx, ]
        pred_mask_27 = cpreds_27[idx, ]
        pred_mask_62 = cpreds_62[idx, ]
        pred_mask_378 = cpreds_378[idx, ]
        chamber_size_20, chamber_limits_20, max_chamber_height_20 = calc_img_chamber_size(pred_mask_20)
        chamber_size_27, chamber_limits_27, max_chamber_height_27 = calc_img_chamber_size(pred_mask_27)
        chamber_size_62, chamber_limits_62, max_chamber_height_62 = calc_img_chamber_size(pred_mask_62)
        chamber_size_378, chamber_limits_378, max_chamber_height_378 = calc_img_chamber_size(pred_mask_378)

        plt.figure(1)
        plt.clf()
        plt.subplot(421)
        plt.imshow(ds_img)
        plt.subplot(422)
        plt.imshow(pred_mask_50)
        plt.scatter(x=chamber_limits_50[:,1], y=chamber_limits_50[:,0], c='red', s=1)
        plt.subplot(423)
        plt.imshow(pred_mask_88)
        plt.scatter(x=chamber_limits_88[:,1], y=chamber_limits_88[:,0], c='red', s=1)
        plt.subplot(424)
        plt.imshow(pred_mask_99)
        plt.scatter(x=chamber_limits_99[:,1], y=chamber_limits_99[:,0], c='red', s=1)

        plt.subplot(425)
        plt.imshow(pred_mask_20)
        plt.scatter(x=chamber_limits_20[:, 1], y=chamber_limits_20[:, 0], c='red', s=1)
        plt.subplot(426)
        plt.imshow(pred_mask_27)
        plt.scatter(x=chamber_limits_27[:, 1], y=chamber_limits_27[:, 0], c='red', s=1)
        plt.subplot(427)
        plt.imshow(pred_mask_62)
        plt.scatter(x=chamber_limits_62[:, 1], y=chamber_limits_62[:, 0], c='red', s=1)
        plt.subplot(428)
        plt.imshow(pred_mask_378)
        plt.scatter(x=chamber_limits_378[:, 1], y=chamber_limits_378[:, 0], c='red', s=1)

        base_name = parse_img_base_name(pname)
        plt.savefig('{}/{}'.format('./accell/seg_debug2', base_name), bbox_inches='tight')
    return


# HACK to get best chambers for generating ac cell data
def get_best_chamber_preds():
    seg_img_names, ds_imgs_seg, raw_imgs_seg = get_accell_img_data(mode='segmented')
    weight_files = ['weights-improvement-378--0.91209759.hdf5', 'weights-improvement-027--0.95307544.hdf5',
                    'weights-improvement-020--0.94153012.hdf5', 'weights-improvement-062--0.96126655.hdf5']

    for idx, weight_file in enumerate(weight_files):
        if idx==0:
            weights_folder = '/home/yue/pepple/runs/2017-08-09-19-02-27/weights'
            lr = '_slow'
        else:
            weights_folder = '/home/yue/pepple/runs/2017-08-09-10-20-24/weights'
            lr = ''

        epoch_num = int(weight_file.split('-')[2])
        all_preds_path = 'all_preds_{}{}.npy'.format(epoch_num, lr)
        predict_chamber(ds_imgs_seg, all_preds_path, chamber_weights_folder=weights_folder, test_weights=weight_file)
    return


def remove_bad_chamber_seg_files():
    bad_file_path = '{}/{}'.format('./accell', 'bad_chamber_seg.txt')
    bad_files = []
    with open(bad_file_path) as fin:
        for l in fin:
            bad_files.append(l.rstrip())

    seg_img_names, ds_imgs_seg, raw_imgs_seg = get_accell_img_data(mode='segmented')
    base_names = [parse_img_base_name(x) for x in seg_img_names]
    common_files = set(bad_files).intersection(base_names)
    len(bad_files)
    len(common_files)

    # open coord file and re-write to clean coord file
    outfile = '{}/{}.txt'.format(augmented_folder_master, 'training_coords')
    path_prefix = '/home/yue/pepple/accell/augmented/'
    clean_lines = []
    with open(outfile) as fin:
        for l in fin:
            arr = l.rstrip().split(",")  # faster-rcnn input format
            file_name = arr[0].replace(path_prefix, '')
            base_name = '_'.join(file_name.split('_')[:5])
            if base_name not in bad_files:
                clean_lines.append(l)
    outfile2 = '{}/{}.txt'.format(augmented_folder_master, 'training_coords_clean')
    with open(outfile2, 'w') as fout:
        for l in clean_lines:
            fout.write('{}'.format(l))

    outfile = '{}/{}.txt'.format(augmented_folder_master, 'test_coords')
    path_prefix = '/home/yue/pepple/accell/augmented_test/'
    clean_lines = []
    with open(outfile) as fin:
        for l in fin:
            arr = l.rstrip().split(",")  # faster-rcnn input format
            file_name = arr[0].replace(path_prefix, '')
            base_name = '_'.join(file_name.split('_')[:5])
            if base_name not in bad_files:
                clean_lines.append(l)
    outfile2 = '{}/{}.txt'.format(augmented_folder_master, 'test_coords_clean')
    with open(outfile2, 'w') as fout:
        for l in clean_lines:
            fout.write('{}'.format(l))

    return


# samples patches from raw images instead
def img_sampler_helper(img_mean, img_std, num_rows, num_cols, visualise=False):
    num_imgs = SAMPLER_CLEANED_IMGS.shape[0]
    sample_idx = random.randint(0, num_imgs-1)

    cur_preds = SAMPLER_PREDS[sample_idx,]
    pred_threshold = 0.9  # very conservative threshold
    chamber_limits = np.argwhere(cur_preds > pred_threshold)  # n*2 where either 1st/2nd coord below pred_threshold
    chamber_intensities = SAMPLER_CLEANED_IMGS[sample_idx, chamber_limits[:, 0], chamber_limits[:, 1], 0]

    # mean_y, mean_x = np.mean(chamber_limits, axis=0)    # careful about meaning of coordinates: 1st coord is y-axis
    mean_x = np.mean(chamber_limits[:, 1])
    # mean_y = np.percentile(chamber_limits[:, 0], q=25)
    mean_y = np.mean(chamber_limits[chamber_limits[:, 1] == int(mean_x), 0])  # mid-y for central x

    box_idx = random.randint(0, chamber_limits.shape[0] - 1)  # random point inside mask
    cur_coord = chamber_limits[box_idx,]
    cur_y, cur_x = cur_coord
    # move towards image center
    move_factor = 0.5  # allow some chamber border cases; 1 for less chamber border cases
    move_factor = 1.0  # no borders as thats confusing
    cur_x = int(cur_x + move_factor * num_cols // 2 * np.sign(mean_x - cur_x))
    cur_y = int(cur_y + move_factor * num_rows // 2 * np.sign(mean_y - cur_y))

    x1 = int(cur_x * DOWNSAMPLE_RATIO - num_cols // 2)  # rescale
    x2 = int(cur_x * DOWNSAMPLE_RATIO + num_cols // 2)
    y1 = int(cur_y * DOWNSAMPLE_RATIO - num_rows // 2)
    y2 = int(cur_y * DOWNSAMPLE_RATIO + num_rows // 2)
    cur_patch = SAMPLER_CLEANED_IMGS[sample_idx, y1:y2, x1:x2, 0]

    chamber_mean = np.mean(chamber_intensities)
    chamber_std = np.std(chamber_intensities)
    # if np.mean(cur_patch) < chamber_mean+chamber_std:
    if np.mean(cur_patch) < max(chamber_mean+5, 40):    # if mean_std=pixel_std/sqrt(32)
        return cur_patch, cur_preds, sample_idx, box_idx, cur_coord, chamber_limits, chamber_mean, chamber_std, x1, x2, y1, y2
    else:
        return img_sampler_helper(img_mean, img_std, num_rows, num_cols, visualise)


def img_sampler_helper2(img_mean, img_std, num_rows, num_cols, visualise=False):
    num_imgs = SAMPLER_CLEANED_IMGS.shape[0]
    sample_idx = random.randint(0, num_imgs-1)

    cur_preds = SAMPLER_PREDS[sample_idx, ]
    pred_threshold = 0.9  # very conservative threshold
    # chamber_limits y_coord listed first
    chamber_limits = np.argwhere(cur_preds > pred_threshold)  # n*2 where either 1st/2nd coord below pred_threshold
    chamber_intensities = SAMPLER_CLEANED_IMGS[sample_idx, chamber_limits[:, 0], chamber_limits[:, 1], 0]

    box_idx = random.randint(0, chamber_limits.shape[0] - 1)  # random point inside mask
    cur_coord = chamber_limits[box_idx, ]
    cur_y, cur_x = cur_coord
    x1 = int(cur_x - num_cols // (2 * DOWNSAMPLE_RATIO))  # original scale
    x2 = int(cur_x + num_cols // (2 * DOWNSAMPLE_RATIO))  # original scale
    y1 = int(cur_y - num_cols // (2 * DOWNSAMPLE_RATIO))  # original scale
    y2 = int(cur_y + num_cols // (2 * DOWNSAMPLE_RATIO))  # original scale

    # x1 = int(cur_x * DOWNSAMPLE_RATIO - num_cols // 2)  # rescale
    # x2 = int(cur_x * DOWNSAMPLE_RATIO + num_cols // 2)
    # y1 = int(cur_y * DOWNSAMPLE_RATIO - num_rows // 2)
    # y2 = int(cur_y * DOWNSAMPLE_RATIO + num_rows // 2)

    # check all 4 corners in chamber limits
    # FIXME - possible 4 corners in limits, but other parts not
    corners = [(y1, x1), (y2, x1), (y1, x2), (y2, x2)]  # careful of order in chamber_limits
    all_in = True
    for corner in corners:
        corner_idx = np.where((chamber_limits == corner).all(axis=1))[0]
        all_in = all_in and len(corner_idx) > 0

    x1 = int(x1 * DOWNSAMPLE_RATIO)
    x2 = int(x2 * DOWNSAMPLE_RATIO)
    y1 = int(y1 * DOWNSAMPLE_RATIO)
    y2 = int(y2 * DOWNSAMPLE_RATIO)
    cur_patch = SAMPLER_CLEANED_IMGS[sample_idx, y1:y2, x1:x2, 0]

    # FIXME absolute intensity condition but ideally should be - if np.mean(cur_patch) > chamber_mean+chamber_std/sqrt(32):
    if all_in and np.mean(cur_patch) < 25+15:   # 2nd condition in case bad segmentation
        chamber_mean = np.mean(chamber_intensities)
        chamber_std = np.std(chamber_intensities)

        if visualise:
            plt.figure(1)
            plt.clf()
            plt.imshow(cur_preds)
            plt.scatter(x=cur_coord[1], y=cur_coord[0], c='red', s=1)

            plt.figure(2)
            plt.clf()
            plt.imshow(SAMPLER_CLEANED_IMGS[sample_idx, :, :, 0])
            plt.scatter(x=chamber_limits[:, 1]*DOWNSAMPLE_RATIO, y=chamber_limits[:, 0]*DOWNSAMPLE_RATIO, c='green', s=1)
            plt.scatter(x=[x1, x2, x1, x2], y=[y1, y2, y2, y1], c='red', s=2)
        return cur_patch, cur_preds, sample_idx, box_idx, cur_coord, chamber_limits, chamber_mean, chamber_std, x1, x2, y1, y2
    else:
        return img_sampler_helper2(img_mean, img_std, num_rows, num_cols, visualise)



def img_sampler(img_mean, img_std, num_rows, num_cols, visualise=False):
    num_imgs = SAMPLER_CLEANED_IMGS.shape[0]
    cur_patch, cur_preds, sample_idx, box_idx, cur_coord, chamber_limits, chamber_mean, chamber_std, x1, x2, y1, y2 \
        = img_sampler_helper2(img_mean, img_std, num_rows, num_cols, visualise)

    # bad patch
    # if np.mean(cur_patch) > chamber_mean+chamber_std:
    if np.mean(cur_patch) > 25+15:
        print('possible bad embedding patch; chamber_mean:{}; chamber_std:{}; patch_mean:{}; patch_std:{}'
            .format(chamber_mean, chamber_std, np.mean(cur_patch), np.std(cur_patch)))

    if visualise:
        plt.figure(2)   # scaled prediction mask
        plt.clf()
        mean_x = np.mean(chamber_limits[:, 1])
        mean_y = np.mean(chamber_limits[chamber_limits[:, 1] == int(mean_x), 0])  # mid-y for central x
        plt.imshow(SAMPLER_PREDS[sample_idx, ])
        plt.scatter(x=mean_x, y=mean_y, c='blue', s=2)
        plt.scatter(x=cur_coord[1], y=cur_coord[0], c='red', s=2)

        plt.figure(1)   # original scale image
        plt.clf()
        plt.imshow(SAMPLER_CLEANED_IMGS[sample_idx, :, :, 0])
        plt.scatter(x=chamber_limits[:, 1] * 2, y=chamber_limits[:, 0] * 2, c='green', s=1)
        plt.scatter(x=mean_x*DOWNSAMPLE_RATIO, y=mean_y*DOWNSAMPLE_RATIO, c='blue', s=2)
        plt.scatter(x=cur_coord[1]*DOWNSAMPLE_RATIO, y=cur_coord[0]*DOWNSAMPLE_RATIO, c='red', s=2)
        plt.scatter(x=[x1, x1, x2, x2], y=[y1, y2, y1, y2], c='pink', s=2)

        plt.figure(3)
        plt.clf()
        plt.imshow(cur_patch)
        1

        # visualise all imgs vs preds for good images - to double check
        if 0:
            for idx in range(num_imgs):
                cur_preds = SAMPLER_PREDS[idx,]
                cur_orig = SAMPLER_CLEANED_IMGS[idx, :, :, 0]
                plt.figure(2); plt.clf()
                plt.imshow(cur_preds)
                plt.figure(1); plt.clf()
                plt.imshow(cur_orig)
                pred_threshold = .9
                chamber_limits = np.argwhere(cur_preds > pred_threshold)
                plt.scatter(x=chamber_limits[:, 1]*DOWNSAMPLE_RATIO, y=chamber_limits[:, 0]*DOWNSAMPLE_RATIO, c='red', s=1)
                # mean_y, mean_x = np.mean(chamber_limits, axis=0)
                mean_x = np.mean(chamber_limits[:, 1])
                # mean_y = np.percentile(chamber_limits[:, 0], q=25)
                mean_y = np.mean(chamber_limits[chamber_limits[:, 1] == int(mean_x), 0])  # mid-y for central x
                plt.scatter(x=mean_x*DOWNSAMPLE_RATIO, y=mean_y*DOWNSAMPLE_RATIO, c='blue', s=2)
                plt.title(SAMPLER_IMG_NAMES[idx])
                1
                # input("Press Enter to continue...")

    return cur_patch


def bad_seg_file_names():
    bad_file_path = '{}/{}'.format('./accell', 'bad_chamber_seg.txt')
    bad_files = []
    with open(bad_file_path) as fin:
        for l in fin:
            bad_files.append(l.rstrip())
    return bad_files


def img_generator(img_mean, img_std, num_rows, num_cols):
    # create blank/noisy background
    adj_factor = 1.5 if NOISIER else 3.0
    img = np.round(np.random.normal(img_mean, img_std/adj_factor, (num_rows, num_cols)))
    img[img < 0] = 1
    return img


# ac cells on blank/background noise data
def create_blank_ac_cell_data(num_rows=32, num_cols=32, num_samples=10000, img_fun=img_generator, visualise=False):
    seg_img_names, ds_imgs_seg, raw_imgs_seg = get_accell_img_data(mode='segmented')
    all_cell_data = get_all_ac_cells(seg_img_names, raw_imgs_seg, threshold=0, visualise=False)
    img_mean = np.mean(ds_imgs_seg)
    img_std = np.std(ds_imgs_seg)

    if platform == "linux" or platform == "linux2":
        # base_folder = '/home/yue/pepple/accell/blank_{}_{}{}'.format(num_rows, num_cols, '_noisy' if NOISIER else '')
        base_folder = '/home/yue/pepple/accell/blank_{}_{}{}'.format(num_rows, num_cols, '_realistic2' if NOISIER else '')
    else:
        # base_folder = './accell/blank_{}_{}{}'.format(num_rows, num_cols, '_noisy' if NOISIER else '')
        base_folder = './accell/blank_{}_{}{}'.format(num_rows, num_cols, '_realistic2' if NOISIER else '')

    train_folder = '{}/train'.format(base_folder)
    valid_folder = '{}/valid'.format(base_folder)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        os.makedirs(train_folder)
        os.makedirs(valid_folder)

    train_valid_split = .8
    num_coords_train = int(len(all_cell_data) * train_valid_split)
    cell_train = all_cell_data[:num_coords_train, ]

    for idx in range(num_samples):
        cur_img = img_fun(img_mean, img_std, num_rows=num_rows, num_cols=num_cols)
        cur_img, coords, mid_coords = create_blank_ac_cell_img(cell_train, cur_img)

        img_name = '{}/training_{}.png'.format(train_folder, idx)
        cv2.imwrite(img_name, cur_img)

        if visualise:
            plt.imshow(cur_img)
            plt.scatter(x=mid_coords[:, 0], y=mid_coords[:, 1], c='red', s=1)

        with open('{}/training_coords.txt'.format(base_folder), 'a') as fout:
            for coord in coords:
                vals = [img_name, str(coord[0]), str(coord[1]), str(coord[2]), str(coord[3]), 'cell']
                # print(','.join(vals))
                fout.write('{}\n'.format(','.join(vals)))

    # validation data
    num_test = int(num_samples * .1)
    cell_valid = all_cell_data[num_coords_train:, ]
    for idx in range(num_test):
        cur_img = img_fun(img_mean, img_std, num_rows=num_rows, num_cols=num_cols)
        cur_img, coords, mid_coords = create_blank_ac_cell_img(cell_train, cur_img)

        img_name = '{}/test_{}.png'.format(valid_folder, idx)
        cv2.imwrite(img_name, cur_img)

        with open('{}/valid_coords.txt'.format(base_folder), 'a') as fout:
            for coord in coords:
                vals = [img_name, str(coord[0]), str(coord[1]), str(coord[2]), str(coord[3]), 'cell']
                # print(','.join(vals))
                fout.write('{}\n'.format(','.join(vals)))
    return


def create_blank_ac_cell_img(cell_data, img, visualise=False):
    num_patches, _, _ = cell_data.shape
    num_rows, num_cols = img.shape
    raw_img = np.copy(img)

    # add cells
    max_cells = num_rows*num_cols / (cell_size * cell_size)
    cell_upper = int(max_cells * .3*.5)     # up to 32*32/25*.3*.5 = 12*.5 = 6 cells per image
    cell_lower = int(max_cells * .1*.25)    # at least 1 cell in image for useful training
    num_samples = random.randint(cell_lower, cell_upper)

    mean_x = num_cols/2.
    mean_y = num_rows/2.
    mid_coords = np.zeros((0, 2), dtype=np.uint8)  # pre-allocate
    coords = np.zeros((0, 4), dtype=np.uint8)  # pre-allocate
    for idx in range(num_samples):
        rand_ind = random.choice(range(0, num_patches))
        rand_cell = cell_data[rand_ind, ]

        # random coord in img
        cur_y, cur_x = random.randint(0, num_rows-1), random.randint(0, num_cols-1)
        # move towards img center
        # new_x = int(cur_x + np.ceil(cell_size/2) * np.sign(mean_x - cur_x))
        # new_y = int(cur_y + np.ceil(cell_size/2) * np.sign(mean_y - cur_y))

        # border conditions
        new_x = cur_x
        new_y = cur_y
        if cur_x <= cell_size/2:
            new_x = int(cell_size/2)+1
        elif cur_x >= num_cols - cell_size/2:
            new_x = int(num_cols - cell_size/2 -1)
        if cur_y <= cell_size/2:
            new_y = int(cell_size/2) +1
        elif cur_y >= num_rows - cell_size/2:
            new_y = int(num_rows - cell_size/2 -1)

        # avoid overlapping
        mid_coord = (new_x, new_y)
        overlapping_new = is_overlapping(mid_coords, mid_coord)
        if not overlapping_new:
            # set patch
            x_lower, y_lower, x_upper, y_upper = make_box_coords(new_x, new_y, img_shape=img.shape)
            new_coord = x_lower, y_lower, x_upper, y_upper
            # FIXME - blur cell into background
            # double check cells have been centered
            # img = set_patch(img, rand_cell, (x_lower, y_lower, x_upper, y_upper))
            if img[y_lower:y_upper, x_lower:x_upper].shape != (cell_size, cell_size):
                print('cur_x={}; cur_y={}; new_x={}; new_y={}; new_coord={}'.format(cur_x, cur_y, new_x, new_y, new_coord))
            img = set_cell(img, rand_cell, (x_lower, y_lower, x_upper, y_upper), blur=False)
            mid_coords = np.append(mid_coords, np.reshape(np.asarray(mid_coord), (1, 2)), axis=0)
            coords = np.append(coords, np.reshape(np.asarray(new_coord), (1, 4)), axis=0)

            if visualise:
                plt.figure(1)
                plt.imshow(raw_img)
                plt.scatter(x=new_x, y=new_y, c='red', s=2)  # coord center
                plt.axes().add_patch(patches.Rectangle((x_lower, y_lower), cell_size, cell_size, fill=False, color='green'))   # show box

                plt.figure(3)
                plt.imshow(img)
                plt.scatter(x=new_x, y=new_y, c='red', s=2)  # coord center
                plt.axes().add_patch(patches.Rectangle((x_lower, y_lower), cell_size, cell_size, fill=False, color='green'))   # show box

    return img, coords, mid_coords


def review_blank_accell_data(folder):
    true_coords_dict = get_true_coords(folder, coord_file='training_coords.txt')
    from analyseCellPreds import patch_stats

    img_folder = os.path.join(folder, 'train')
    img_files = os.listdir(img_folder)
    for img_file in img_files:
        img = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_GRAYSCALE)
        img_coords = true_coords_dict[img_file]
        plot_img_boxes(img_file, img, img_coords)
        print(patch_stats(img))
    return


def get_true_coords(folder, coord_file='training_coords.txt'):
    true_dict = {}

    with open(os.path.join(folder, coord_file)) as fin:
        for l in fin:
            arr = l.rstrip().split(",")  # faster-rcnn input format
            file_name = arr[0].split('/')[-1]
            cur_coord = [tuple([int(a) for a in arr[1:-1]])]
            if file_name in true_dict:
                true_dict[file_name] += cur_coord
            else:
                true_dict[file_name] = cur_coord
    return true_dict


def plot_img_boxes(key, img, box_data):
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    for box_coords in box_data:
        x1, y1, x2, y2 = box_coords
        if y1 > RAW_IMAGE_HEIGHT or y2 > RAW_IMAGE_HEIGHT:
            print('y sizing issue for {}; coords={}'.format(key, box_coords))
        if x1 > RAW_IMAGE_WIDTH or x2 > RAW_IMAGE_WIDTH:
            print('y sizing issue for {}; coords={}'.format(key, box_coords))
        ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='white', linewidth=1))
    return


# scale using imagemagick, predict on scaled, cubic interpolate back to original
def predict_raw_imgs():
    predict_chamber()
    return


if __name__ == '__main__':
    # 1st iteration
    # make_test_images()
    # check_json_consistency()

    ## with augmented data
    # make_augmented_image_data()
    # make_test_images(is_aug=2, nrows=320, ncols=128)  # make original test data for ac cells (different image sizes)
    # create_blank_ac_cell_data(num_rows=32, num_cols=32, num_samples=10000)
    # create_blank_ac_cell_data(num_rows=320, num_cols=128, num_samples=10000)
    # create_blank_ac_cell_data(num_rows=128, num_cols=128, num_samples=10000)

    # review_blank_accell_data(folder=os.path.join('accell', 'blank_32_32'))
    # give it more realistic background
    raw_filename = '{}/raw_images_clean.npy'.format(augmented_folder_master)
    SAMPLER_CLEANED_IMGS = np.load(raw_filename)
    bad_seg_files = bad_seg_file_names()
    # bad_seg_base = ['_'.join(x.split('_')[1:]) for x in bad_seg_files]
    # pnames = sorted(list(glob.glob('{}/*.png'.format(img_folder))))
    pnames = sorted(os.listdir(img_folder))
    pnames = [x for x in pnames if '.png' in x and 'mask' not in x]
    # pnames_base = ['_'.join(x.split('_')[1:]) for x in pnames]
    good_seg_idx = []
    for idx, pname in enumerate(pnames):
        if pname.replace('.png', '') not in bad_seg_files:
            good_seg_idx.append(idx)
            SAMPLER_IMG_NAMES.append(pname)
    SAMPLER_CLEANED_IMGS = SAMPLER_CLEANED_IMGS[good_seg_idx, ]  # remove bad segs
    combined_preds_file = 'combined_preds.npy'
    combined_preds_path = '{}/{}'.format(augmented_folder_master, combined_preds_file)
    SAMPLER_PREDS = np.load(combined_preds_path)
    SAMPLER_PREDS = SAMPLER_PREDS[good_seg_idx, :]  # remove bad segs

    create_blank_ac_cell_data(num_rows=32, num_cols=32, num_samples=50000, img_fun=img_sampler)

    # get_best_chamber_preds()
    # debug_chamber_segmentations()   # check chambers segmented properly
    # remove_bad_chamber_seg_files()