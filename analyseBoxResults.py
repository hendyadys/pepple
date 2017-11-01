import numpy as np
import glob, cv2
from ast import literal_eval

import matplotlib.pyplot as plt
from matplotlib import patches
# plt.switch_backend('agg')
from make_bbox_data import downsample, DOWNSAMPLE_RATIO

from sys import platform
if platform == "linux" or platform == "linux2":
    base_folder = './'
    # test_folder = './accell/test'
    # test_figs = './accell/predicted'

    test_folder = './accell/augmented_test'
    test_figs = './accell/augmented_predicted'
    orig_img_folder = './accell/segmentations'
    augmented_folder_master = '/home/yue/pepple/accell/augmented_master'
elif platform == "win32":
    base_folder = './'
    # test_folder = './accell/test'
    # test_figs = './accell/predicted'
    test_folder = './accell/augmented_test'
    # test_folder = './accell/augmented_test_old'
    test_figs = './accell/augmented_predicted'
    orig_img_folder = './accell/segmentations'
    augmented_folder_master = './accell/augmented_master'


def compute_metrics(visualise=False):
    predicted_dict = get_predicted_coords()
    true_dict = get_true_coords()

    test_images = sorted(glob.glob('{0}/*.png'.format(test_folder)))
    total_images = len(test_images)
    strips_per_original_image = 165
    # true number, predicted number, min_dist
    total_score = np.zeros((int(total_images/strips_per_original_image), strips_per_original_image, 3))
    img_count = 0
    for idx, test_image in enumerate(test_images):
        base_name = test_image.replace('{}\\'.format(test_folder), '')

        true_coords = []
        if base_name in true_dict:
            true_coords = true_dict[base_name]
        cur_true = len(true_coords)

        pred_coords = []
        if base_name in predicted_dict:
            pred_coords = predicted_dict[base_name]
        cur_predicted = len(pred_coords)

        min_dist, min_dist_idx = calc_min_dist(true_coords, pred_coords)
        img_count = idx // strips_per_original_image
        img_rem = idx % strips_per_original_image
        total_score[img_count, img_rem, ] = [cur_true, cur_predicted, np.nanmean(min_dist)]

        if visualise and true_coords and pred_coords:
            img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
            visualize_preds(img, true_coords, pred_coords)
    return total_score


def visualize_preds(img, true_coords, pred_coords):
    plt.figure(1)
    plt.clf()
    plt.imshow(img)
    plt.title('original strip')

    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    ax1.set_title('true and predicted ac cells')
    for coord in true_coords:    # add cells to image
        (x1, y1, x2, y2) = coord
        ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='white', linewidth=1))

    for coord in pred_coords:    # add cells to image
        (x1, y1, x2, y2) = coord
        ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=1))
        ax1.scatter(x=[(x1+x2)/2], y=[(y1+y2)/2], c='yellow', s=2)

    # show high intensity parts
    plt.figure(3)
    plt.clf()
    plt.imshow(img)
    high_intensity_points = np.argwhere(img > 60)
    plt.scatter(x=high_intensity_points[:, 1], y=high_intensity_points[:, 0], c='red', s=1)

    return


def calc_min_dist(true_coords, pred_coords):
    min_dist = []
    min_dist_idx = []
    for true_coord in true_coords:
        coord_center = get_box_center(true_coord)
        # cur_min_dist = float('inf')
        cur_min_dist = np.nan
        cur_min_idx = np.nan

        for idx, pred_coord in enumerate(pred_coords):
            pred_center = get_box_center(pred_coord)
            coord_dist = euclid_dist(coord_center, pred_center)

            if np.isnan(cur_min_dist) or coord_dist < cur_min_dist:
                cur_min_dist = coord_dist
                cur_min_idx = idx

        min_dist.append(cur_min_dist)
        if cur_min_dist < 3:    # only add idx if really close to matching
            min_dist_idx.append(cur_min_idx)

    return min_dist, min_dist_idx


def get_box_center(box_coords):
    x1, y1, x2, y2 = box_coords
    return ((x1+x2)/2., (y1+y2)/2.)


def euclid_dist(coord_1, coord_2):
    x_1, y_1 = coord_1
    x_2, y_2 = coord_2

    euclidean_dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
    return euclidean_dist


def get_true_coords(folder=test_folder, get_test_data=True):
    true_dict = {}
    if get_test_data:
        path_prefix = '/home/yue/pepple/accell/augmented_test/'
        path_prefix = '/home/yue/pepple/accell/blank_32_32_noisy/valid/'
        # coord_file = "{}/test_coords_clean.txt".format(folder)
        coord_file = "{}/test_coords.txt".format(folder)
    else:
        path_prefix = '/home/yue/pepple/accell/augmented/'
        path_prefix = '/home/yue/pepple/accell/blank_128_128/train/'
        # folder = './accell/augmented_master'
        # coord_file = "{}/training_coords_clean.txt".format(folder)
        coord_file = "{}/training_coords.txt".format(folder)

    with open(coord_file) as fin:
        for l in fin:
            arr = l.rstrip().split(",")  # faster-rcnn input format
            file_name = arr[0].replace(path_prefix, '')
            cur_coord = [tuple([int(a) for a in arr[1:-1]])]
            if file_name in true_dict:
                true_dict[file_name] += cur_coord
            else:
                true_dict[file_name] = cur_coord
    return true_dict


def get_predicted_coords(folder=test_folder):
    predicted_dict = {}
    # with open("{}/coords.txt".format(results_folder)) as fin:
    # with open("{}/coords.txt".format(folder)) as fin:
    with open("{}/coords_noisy.txt".format(folder)) as fin:
        for l in fin:
            arr = l.rstrip().split("\t")
            file_name = arr[0]
            if file_name in predicted_dict:
                predicted_dict[file_name] += literal_eval(arr[1])
            else:
                predicted_dict[file_name] = literal_eval(arr[1])
    return predicted_dict


# compare predicted vs true and plot where necessary
def compute_metrics2(results_folder=test_folder, do_test=True, visualise=False):
    predicted_dict = get_predicted_coords(folder=results_folder)
    true_dict = get_true_coords(folder=results_folder, get_test_data=do_test)

    # num_true, num_pred, min_dist_t, min_dist_p, intensity_t, intensity_p, intensity_m, num_pred_within_4pixel
    total_score = np.ndarray((len(predicted_dict), 8), dtype=np.float32)
    total_predicted = np.ndarray((len(predicted_dict), 2), dtype=np.float32)
    counter = 0
    all_counter = 0
    for key, value in predicted_dict.items():

        pred_coords = value
        num_preds = len(pred_coords)
        total_predicted[all_counter, 1] = num_preds
        all_counter +=1

        true_coords = []
        if key in true_dict:
            true_coords = true_dict[key]
            num_true = len(true_coords)
            total_predicted[all_counter-1, 0] = num_true
        else:
            total_predicted[all_counter-1, 0] = 0
            continue
        min_dist_t, min_idx_t = calc_min_dist(true_coords, pred_coords)
        min_dist_p, min_idx_p = calc_min_dist(pred_coords, true_coords)
        # are we only identifying brighter cells?
        # img_name = '{}\{}'.format(test_folder, key)
        img_name = '{}\{}'.format(results_folder, key)
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        true_avg_intensity, pred_avg_intensity, matched_avg_intensities = found_vs_missed_analysis(img, true_coords, pred_coords)

        total_score[counter, ] = [num_true, num_preds, np.nanmean(min_dist_t), np.nanmean(min_dist_p),
                                  true_avg_intensity, pred_avg_intensity, np.mean(matched_avg_intensities),
                                  len(matched_avg_intensities)]
        counter += 1

        if visualise and true_coords and pred_coords:
            visualize_preds(img, true_coords, pred_coords)
            1
            # img_bright_spots(img, intensity_threshold=75)

    # some analysis
    analyse_pred_metrics2(total_score, total_predicted, counter)

    return total_score


def img_bright_spots(img, intensity_threshold=50):
    # plt.figure(1)
    # plt.imshow(img)
    img2 = np.argwhere(img>intensity_threshold)
    plt.figure(2)
    plt.clf()
    plt.imshow(img)
    plt.scatter(y=img2[:,0], x=img2[:,1], c='red', s=1)
    return


def found_vs_missed_analysis(img, true_coords, pred_coords):
    true_coords = np.asarray(true_coords)
    pred_coords = np.asarray(pred_coords)

    # compare predicted vs true
    true_patches, true_mean_intensities, true_avg_intensity = get_coord_patches(img, true_coords)
    pred_patches, pred_mean_intensities, pred_avg_intensity = get_coord_patches(img, pred_coords)
    print('found vs missed', true_avg_intensity, pred_avg_intensity)

    # compute patches predicted vs missed
    # assuming more true than predicted
    min_dist_p, min_idx_p = calc_min_dist(pred_coords, true_coords)
    matched_patches = []
    matched_mean_intensities = []
    for idx, patch in enumerate(true_patches):
        if idx in min_idx_p:
            # matched_patches.append(patch)
            matched_mean_intensities.append(true_mean_intensities[idx])
        # matched_avg_intensity = true_avg_intensity[min_idx_p,]

    return true_avg_intensity, pred_avg_intensity, matched_mean_intensities


def check_aug_data(check_test=False):
    true_dict = get_true_coords(get_test_data=check_test)

    for key, value in true_dict.items():
        if check_test:
            img_name = '{}\{}'.format(test_folder, key)
        else:
            img_name = '{}\{}'.format('./accell/augmented', key)
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        visualize_preds(img, value, value)
        # compare cell intensity vs image intensity
        cell_patches, patch_mean_intensities, avg_patch_intensity = get_coord_patches(img, value)
        img_mean_intensity = np.mean(img)
        print('check aug data', avg_patch_intensity, img_mean_intensity, patch_mean_intensities)
        # get_original_img(key, coords=value)
    return


def get_original_img(img_name, coords):
    # double check by loading image and numpy of saved images
    pnames = sorted(list(glob.glob('{}/*.png'.format(orig_img_folder))))
    ds_filename = '{}/downsampled_images.npy'.format(augmented_folder_master)
    ds_data = np.load(ds_filename)
    # raw_filename = '{}/raw_images.npy'.format(augmented_folder_master)
    # raw_data = np.load(raw_filename)

    img_toks = img_name.split('_')
    img_base_name = '_'.join(img_toks[:-2])
    x_shift = int(img_toks[-2])
    y_shift = int(img_toks[-1].replace('.png', ''))
    idx = pnames.index('{}\{}.png'.format(orig_img_folder, img_base_name))
    # visualize
    ds_img = ds_data[int(idx/2), :, :, 0]
    adj_coords = [(x[0]+x_shift, x[1]+y_shift, x[2]+x_shift, x[3]+y_shift) for x in coords]
    visualize_preds(ds_img, adj_coords, adj_coords)
    cell_patches, patch_mean_intensities, avg_patch_intensity = get_coord_patches(ds_img, adj_coords)

    # load raw image and downsample for comparison
    img = cv2.imread('{}/{}.png'.format(orig_img_folder, img_base_name), cv2.IMREAD_GRAYSCALE)
    img_shape = img.shape
    ds_img2 = downsample(img.reshape((1, img_shape[0], img_shape[1])), factor=int(DOWNSAMPLE_RATIO))
    # visualize_preds(ds_img2, adj_coords, adj_coords)
    np.sum(ds_img==ds_img2)
    return


def get_coord_patches(img, cell_coords, visualise=False):
    # cell_size = 5
    # num_cells = len(cell_coords)
    # cell_patches = np.zeros((num_cells, cell_size, cell_size), dtype=np.float32)
    cell_patches = []
    patch_mean_intensities = []
    for idx, coord in enumerate(cell_coords):
        x_lower, y_lower, x_upper, y_upper = coord
        cur_patch = img[y_lower:y_upper+1, x_lower:x_upper+1]
        # cell_patches[idx, ] = cur_patch
        cell_patches.append(cur_patch)
        patch_mean_intensities.append(np.nanmean(cur_patch))

        if visualise:
            plt.imshow(img)
            x_mid, y_mid = (.5*(x_lower+x_upper), .5*(y_lower+y_upper))
            plt.scatter(x=x_mid, y=y_mid, c='r', s=3)

        if len(patch_mean_intensities)==0:
            patch_mean_intensities = [0]
    return cell_patches, patch_mean_intensities, np.mean(patch_mean_intensities)


def confirm_coord_with_cone_data():
    cone_dict = get_cone_coords()
    valid_imgs_folder = '../overview/1-process-data/final/valid'
    img_names = sorted(list(glob.glob('{}/*.png'.format(valid_imgs_folder))))
    img_names = ['../overview/1-process-data/final/valid\PrefixS_V011_cropped_master_image_215-51_196-51_83-98_82-98-txt-0-0.png']
    for img_name in img_names:
        if 'mask' in img_name: continue

        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img)
        img_name2 = img_name.replace('../overview/1-process-data/', './')
        img_coords = cone_dict[img_name2]
        img_coords = np.asarray(img_coords)
        plt.scatter(x=0.5*(img_coords[:,0]+img_coords[:,2]), y=0.5*(img_coords[:,1]+img_coords[:,3]), c='red', s=2)
        # plt.scatter(y=0.5*(img_coords[:,0]+img_coords[:,2]), x=0.5*(img_coords[:,1]+img_coords[:,3]), c='blue', s=2)
    return


def confirm_coord_with_pascal_data():
    pascal_img_folder = '../VOCdevkit/VOC2012/JPEGImages'
    pascal_annotations_folder = '../VOCdevkit/VOC2012/Annotations'
    file_names = ['2007_000129']
    for file_name in file_names:
        img_name = '{}/{}.jpg'.format(pascal_img_folder, file_name)
        annot_name = '{}/{}.xml'.format(pascal_annotations_folder, file_name)
        img = cv2.imread(img_name)
        plt.imshow(img)
        import xml.etree.ElementTree as ET
        root = ET.parse(annot_name).getroot()
        img_coords = []
        for obj in root.findall('object'):
            bnd_boxes = obj.findall('bndbox')
            for bnd_box in bnd_boxes:
                xmin = int(bnd_box.find('xmin').text)
                xmax = int(bnd_box.find('xmax').text)
                ymin = int(bnd_box.find('ymin').text)
                ymax = int(bnd_box.find('ymax').text)
                img_coords.append((xmin, ymin, xmax, ymax))
        img_coords2 = np.asarray(img_coords)
        # plt.scatter(x=img_coords2[:, 0], y=img_coords2[:, 1], c='red', s=2)
        # plt.scatter(x=img_coords2[:, 2], y=img_coords2[:, 3], c='blue', s=2)
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img)
        for coord in img_coords:  # add cells to image
            (x1, y1, x2, y2) = coord
            ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='white', linewidth=1))
    return


def get_cone_coords(file='../overview/1-process-data/cone_data_valid.txt'):
    coord_dict = {}
    with open(file) as fin:
        for l in fin:
            arr = l.rstrip().split(",")  # faster-rcnn input format
            img_name = arr[0]
            cur_coord = [tuple([int(a) for a in arr[1:-1]])]
            if img_name in coord_dict:
                coord_dict[img_name] += cur_coord
            else:
                coord_dict[img_name] = cur_coord
    return coord_dict


# def analyse_pred_metrics2(pred_npy='predicted_metrics.npy', ):
def analyse_pred_metrics2(total_score, total_predicted, counter):
    # num_true, num_pred, min_dist_t, min_dist_p, intensity_t, intensity_p, intensity_m
    # total_score = np.load(pred_npy)

    # img_mean = np.nanmean(total_score, axis=0)    # doesn't work if overpredicting counter<len(predicted_dict)
    pred_minus_true = total_predicted[:, 1] - total_predicted[:, 0]
    pred_min_true_quantiles = np.percentile(pred_minus_true, [10, 50, 90])
    false_discovery_rate = np.sum(pred_minus_true) / np.sum(total_predicted[:, 1])  # false_positive/pred_positive
    precision = 1 - false_discovery_rate

    pred_minus_true2 = total_score[range(counter), 1] - total_score[range(counter), 0]
    pred_min_true_quantiles2 = np.percentile(pred_minus_true2, [10, 50, 90])
    score_mean = np.nanmean(total_score[range(counter),], axis=0)

    pred_close = total_score[range(counter), 0] - total_score[range(counter), -1]
    num_missed_preds = np.sum(pred_close[np.where(pred_close > 0),])
    false_neg_rate = num_missed_preds / np.sum(total_score[range(counter), 0])
    sensitivity = 1 - false_neg_rate

    return precision, sensitivity, false_discovery_rate, false_neg_rate, pred_min_true_quantiles, pred_min_true_quantiles2


def analyse_pred_metrics(pred_npy='predicted_metrics.npy'):
    total_score = np.load(pred_npy)
    img_mean = np.nanmean(total_score, axis=1)  # average over strips of image
    img_mean_summary = np.nanmean(total_score, axis=(1, 0))  # average over strips of image
    plt.scatter(img_mean[:, 0], img_mean[:, 1])
    plt.title('number of labeled vs number of predicted cells')
    plt.ylabel('predicted number of cells')
    plt.xlabel('number of labeled cells')
    plt.grid()
    plt.show()

    # average across strips with non-zero targets - number of true vs predicted only when true occurs
    nonzero_label_preds = np.nanmean(total_score[total_score[:, :, 0] != 0,], 0)
    # average across strips with zero targets - for over-prediction
    zero_label_preds = np.nanmean(total_score[total_score[:, :, 0] == 0,], 0)
    1

    # average if prediction nonzero
    # pred_sum = aligned_img.sum(0)
    # pred_nonzero = (aligned_img != 0).sum(0).astype(float)
    # avg_preds = np.true_divide(pred_sum, pred_nonzero)
    # avg_preds[pred_nonzero == 0] = 0
    return


if __name__ == '__main__':
    # check_aug_data(check_test=False)  # check generated data
    # confirm_coord_with_cone_data()    # make sure ac cell data coordinates setup in same way
    # confirm_coord_with_pascal_data()

    ### for cone data
    # total_score = compute_metrics(visualise=True)
    # score_npy = 'predicted_metrics.npy'

    ### ac cell data
    # total_score = compute_metrics2(visualise=False)
    # total_score = compute_metrics2(results_folder='./accell/augmented', do_test=False, visualise=False)
    # total_score = compute_metrics2(results_folder='./accell/blank_32_32/valid', do_test=True, visualise=False)
    # total_score = compute_metrics2(results_folder='./accell/blank_128_128/valid', do_test=True, visualise=False)
    # total_score = compute_metrics2(results_folder='./accell/blank_128_128/train', do_test=False, visualise=False)   # to see if we can overfit
    total_score = compute_metrics2(results_folder='./accell/blank_32_32_noisy/valid', do_test=True, visualise=False)
    score_npy = 'predicted_metrics.npy'
    # np.save(score_npy, total_score)
