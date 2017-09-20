import numpy as np
import glob, cv2
from ast import literal_eval

import matplotlib.pyplot as plt
from matplotlib import patches
# plt.switch_backend('agg')

from sys import platform
if platform == "linux" or platform == "linux2":
    base_folder = './'
    results_folder = './accell/test'
    results_figs = './accell/predicted'
elif platform == "win32":
    base_folder = './'
    base_folder2 = './accell'
    results_folder = './accell/test'
    results_figs = './accell/predicted'


def compute_metrics(visualise=False):
    predicted_dict = get_predicted_coords()
    true_dict = get_true_coords()

    test_images = sorted(glob.glob('{0}/*.png'.format(results_folder)))
    total_images = len(test_images)
    strips_per_original_image = 165
    # true number, predicted number, min_dist
    total_score = np.zeros((int(total_images/strips_per_original_image), strips_per_original_image, 3))
    img_count = 0
    for idx, test_image in enumerate(test_images):
        base_name = test_image.replace('{}\\'.format(results_folder), '')

        true_coords = []
        if base_name in true_dict:
            true_coords = true_dict[base_name]
        cur_true = len(true_coords)

        pred_coords = []
        if base_name in predicted_dict:
            pred_coords = predicted_dict[base_name]
        cur_predicted = len(pred_coords)

        min_dist = calc_min_dist(true_coords, pred_coords)
        img_count = idx // strips_per_original_image
        img_rem = idx % strips_per_original_image
        total_score[img_count, img_rem, ] = [cur_true, cur_predicted, np.nanmean(min_dist)]

        if visualise and true_coords and pred_coords:
            img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
            visualize_preds(img, true_coords, pred_coords, flipAxis=False)
    return total_score


def visualize_preds(img, true_coords, pred_coords, flipAxis=False):
    plt.figure(1)
    plt.clf()
    plt.imshow(img)
    plt.title('original strip')

    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    ax1.set_title('true and predicted ac cells')
    for coord in true_coords:    # add cells to image
        (x1, y1, x2, y2) = coord
        if flipAxis:    # shouldnt need to flip
            ax1.add_patch(patches.Rectangle((y1, x1), y2 - y1, x2 - x1, fill=False, color='red', linewidth=2))
        else:
            ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))  # not flipped

    for coord in pred_coords:    # add cells to image
        (x1, y1, x2, y2) = coord
        if flipAxis:    # shouldnt need to flip
            ax1.add_patch(patches.Rectangle((y1, x1), y2 - y1, x2 - x1, fill=False, color='white', linewidth=2))
        else:
            ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='white', linewidth=2))  # not flipped
            ax1.scatter(x=[(x1+x2)/2], y=[(y1+y2)/2], c='yellow', s=2)
    return


def calc_min_dist(true_coords, pred_coords):
    min_dist = []
    for true_coord in true_coords:
        coord_center = get_box_center(true_coord)
        # cur_min_dist = float('inf')
        cur_min_dist = np.nan

        for pred_coord in pred_coords:
            pred_center = get_box_center(pred_coord)
            coord_dist = euclid_dist(coord_center, pred_center)

            if np.isnan(cur_min_dist) or coord_dist < cur_min_dist:
                cur_min_dist = coord_dist

        min_dist.append(cur_min_dist)

    return min_dist


def get_box_center(box_coords):
    x1, y1, x2, y2 = box_coords
    return ((x1+x2)/2., (y1+y2)/2.)


def euclid_dist(coord_1, coord_2):
    x_1, y_1 = coord_1
    x_2, y_2 = coord_2

    euclidean_dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
    return euclidean_dist


def get_true_coords():
    true_dict = {}
    path_prefix = '/home/yue/pepple/accell/test/'
    with open("{}/test_coords.txt".format(base_folder)) as fin:
        for l in fin:
            arr = l.rstrip().split(",") # faster-rcnn input format
            file_name = arr[0].replace(path_prefix, '')
            cur_coord = [tuple([int(a) for a in arr[1:-1]])]
            if file_name in true_dict:
                true_dict[file_name] += cur_coord
            else:
                true_dict[file_name] = cur_coord
    return true_dict


def get_predicted_coords():
    predicted_dict = {}
    # with open("{}/coords.txt".format(results_folder)) as fin:
    with open("{}/coords.txt".format(base_folder2)) as fin:
        for l in fin:
            arr = l.rstrip().split("\t")
            file_name = arr[0]
            if file_name in predicted_dict:
                predicted_dict[file_name] += literal_eval(arr[1])
            else:
                predicted_dict[file_name] = literal_eval(arr[1])
    return predicted_dict


if __name__ == '__main__':
    # total_score = compute_metrics(visualise=True)
    total_score = compute_metrics(visualise=False)
    score_npy = 'predicted_metrics.npy'
    # np.save(score_npy, total_score)

    # total_score = np.load(score_npy)
    # img_mean = np.mean(total_score, axis=1)
    img_mean = np.nanmean(total_score, axis=1)  # average over strips of image
    img_mean_summary = np.nanmean(total_score, axis=(1, 0))  # average over strips of image
    plt.scatter(img_mean[:,0], img_mean[:,1])
    plt.title('number of labeled vs number of predicted cells')
    plt.ylabel('predicted number of cells')
    plt.xlabel('number of labeled cells')
    plt.grid()
    plt.show()

    # average across strips with non-zero targets - number of true vs predicted only when true occurs
    nonzero_label_preds = np.nanmean(total_score[total_score[:,:,0] != 0,], 0)
    # average across strips with zero targets - for over-prediction
    zero_label_preds = np.nanmean(total_score[total_score[:, :, 0] == 0,], 0)
    1

    # average if prediction nonzero
    # pred_sum = aligned_img.sum(0)
    # pred_nonzero = (aligned_img != 0).sum(0).astype(float)
    # avg_preds = np.true_divide(pred_sum, pred_nonzero)
    # avg_preds[pred_nonzero == 0] = 0