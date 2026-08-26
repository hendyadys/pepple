import numpy as np
import random, os, cv2, json
from matplotlib import pyplot as plt

ACCELL_DIAMETER = 5
DOWNSAMPLE_RATIO = 2

from predict_image_cells import get_mask_boundary
from make_ac_patches import visualise_img_patch
from make_accell_data import find_chamber_center, get_patch_stats, in_ac_chamber, get_img_predictions, avg_images, img_folder


# this takes a thresholding approach to find cells - NOT deep learning!
def get_rolling_means(img, img_pred, find_on_scaled=True, visualise=False):
    img_shape = img.shape
    # overlapping avg can use pandas.rolling_apply, but too lazy to install
    chamber_limits, x1, y1, x2, y2 = get_mask_boundary(img_pred)
    avg_map = np.zeros(shape=img_shape)
    for x in range(x1, x2):
        for y in range(y1, y2):
            if find_on_scaled:
                cell_size=3
            else:
                cell_size = ACCELL_DIAMETER
            y_lower, y_upper = int(y - np.floor(cell_size/2)), int(y+np.ceil(cell_size/2))
            x_lower, x_upper = int(x - np.floor(cell_size/2)), int(x+np.ceil(cell_size/2))
            cur_cell = img[y_lower:y_upper, x_lower:x_upper]    # centered at x, y
            avg_map[y, x] = np.mean(cur_cell)
            if visualise:
                visualise_img_patch(img, [x_lower, y_lower, x_upper, y_upper])
    return avg_map


# create mask - 0 non-AC chamber pixels
def create_mask(raw_img, converted_img, img_pred, find_on_scaled=True, ac_threshold=0.5, visualise=False):
    # ac_threshold = 0.5  # compromise - need to get rid of middle spike!; NB higher number is more conservative

    if find_on_scaled:
        target_img = converted_img
    else:
        target_img = raw_img

    img_shape = target_img.shape
    chamber_area = target_img.copy()
    chamber_limits, mean_x, mean_y = find_chamber_center(img_pred, pred_threshold=ac_threshold)
    non_chamber_limits = np.nonzero(img_pred[0:img_shape[0], 0:img_shape[1]] < ac_threshold)  # for indexing
    chamber_area[non_chamber_limits] = 0  # img_pred is mask. set outside areas to 0

    if visualise:
        chamber_limits, x1, y1, x2, y2 = get_mask_boundary(img_pred, pred_threshold=ac_threshold)  # less conservative for larger ac chamber
        plt.figure(1)
        plt.clf()
        plt.imshow(target_img)
        plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='yellow', s=1)
        plt.figure(2)
        plt.clf()
        plt.imshow(chamber_area)
    return chamber_area, chamber_limits


# thresholding apporach once ac chamber predicted
def threshold_chamber_for_cells(img, scaled_img, img_pred, img_name, find_on_scaled=True, do_overlapping_cells=True,
                                ac_threshold=.5, visualise=False):
    cells = []  # output array

    # zero-out non-ac chamber pixels
    img_mean, img_std, img_shape = get_patch_stats(img)     # not on zero'd put mask
    chamber_area, chamber_limits = create_mask(img, scaled_img, img_pred, find_on_scaled=True, ac_threshold=ac_threshold)

    if do_overlapping_cells:
        chamber_area = get_rolling_means(chamber_area, img_pred, find_on_scaled=find_on_scaled)  # computes local avg intensity

    # threshold
    cell_factor = 1.8    # to be similar to cell definition
    indices = np.argwhere(chamber_area > img_mean*cell_factor)
    if visualise:
        plt.figure(100)
        plt.clf()
        plt.imshow(scaled_img)
        plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='y', s=1)
        plt.figure(1)
        plt.clf()
        plt.imshow(chamber_area)    # this might be smoothed because of do_overlapping_cells
        plt.scatter(x=indices[:, 1], y=indices[:, 0], c='yellow', s=1)
        plt.figure(2)
        plt.clf()
        plt.imshow(img)
        plt.figure(3)
        plt.clf()
        plt.imshow(scaled_img)

    cell_vals = []
    for idx in indices:
        x, y = idx[1], idx[0]   # as in scatter
        cell_vals.append(chamber_area[y, x])
    cell_indices_sorted = np.argsort(cell_vals)[::-1]    # reverse order, ie. descending
    cell_vals_sorted = [cell_vals[x] for x in cell_indices_sorted]
    indices_sorted = np.array([indices[x,] for x in cell_indices_sorted])
    # indices = np.nonzero(chamber_area > img_mean * cell_factor)
    # cell_vals = chamber_area[indices]

    counter = 0
    while len(indices_sorted) > 0:
        y, x = indices_sorted[0]
        cur_cell_val = chamber_area[y, x]
        # get cell for idx
        if find_on_scaled:
            cell_size = 3
        else:
            cell_size = ACCELL_DIAMETER
        y_lower, y_upper = int(y - np.floor(cell_size/ 2)), int(y + np.ceil(cell_size / 2))
        x_lower, x_upper = int(x - np.floor(cell_size / 2)), int(x + np.ceil(cell_size / 2))
        cur_cell = chamber_area[y_lower:y_upper, x_lower:x_upper]
        cell_coord = [x_lower, y_lower, x_upper, y_upper]

        # check if whole cell in chamber
        if find_on_scaled:
            scaled_x = x    # already on rescaled size
            scaled_y = y
        else:
            scaled_x = int(x/DOWNSAMPLE_RATIO)
            scaled_y = int(y/DOWNSAMPLE_RATIO)
        is_in_ac_chamber = in_ac_chamber(scaled_x, scaled_y, chamber_limits, patch_size=cell_size)
        if is_in_ac_chamber:
            cells.append([x, y] + cell_coord)

        if visualise:
            visualise_img_patch(chamber_area, cell_coord)
        # check it is cell  - this is incorporated if do_overlapping_cells
        # check it is maximal local cell - since sorted this doesnt matter

        # now remove neighbouring indices
        indices_sorted = remove_neighbouring_indices(cell_coord, indices_sorted)
        print(counter, 'cell=', idx, len(indices_sorted))
        counter+=1

    if find_on_scaled:
        cells_rescaled = []
        cell_size=ACCELL_DIAMETER
        cell_mid = []
        for cell in cells:
            x, y, x1, y1, x2, y2 = cell
            y_lower, y_upper = int(y*DOWNSAMPLE_RATIO - np.floor(cell_size / 2)), int(y*DOWNSAMPLE_RATIO + np.ceil(cell_size / 2))
            x_lower, x_upper = int(x*DOWNSAMPLE_RATIO - np.floor(cell_size / 2)), int(x*DOWNSAMPLE_RATIO + np.ceil(cell_size / 2))
            cells_rescaled.append([x_lower, y_lower, x_upper, y_upper])
            cell_mid.append([y*DOWNSAMPLE_RATIO, x*DOWNSAMPLE_RATIO])
        cells = np.array(cells_rescaled)
        if visualise:
            visualise_img_patch(img, np.array(cell_mid))
    else:   # TODO - implement for original scale
        cells = np.array(cells)[2:]
        if visualise:
            visualise_img_patch(img, cells)
    return cells.tolist()


def remove_neighbouring_indices(cell_coord, indices_sorted):
    # generate neighbouring cell coordinates
    x1, y1, x2, y2 = cell_coord
    neighbouring_coords = []
    for x in range(x1, x2):
        for y in range(y1, y2):
            neighbouring_coords.append([y, x])  # be consistent with indices_sorted
    neighbouring_coords = np.array(neighbouring_coords).astype('int64')     # enforce same dtype as indices_sorted

    # remove using set diff
    nrows, ncols = neighbouring_coords.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [neighbouring_coords.dtype]}
    # C = np.intersect1d(indices_sorted.view(dtype2), neighbouring_coords.view(dtype1))
    C = np.setdiff1d(indices_sorted.view(dtype), neighbouring_coords.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(neighbouring_coords.dtype).reshape(-1, ncols)
    return C


def threshold_chambers(folder, ac_threshold=0.5, do_avg=True):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)

    cell_dict = {}
    for idx, img_name in enumerate(img_names):
        cur_img = raw_images[idx, ]
        cur_pred = img_preds[idx, ]
        cur_scaled_img = converted_imgs[idx, ]
        cells = threshold_chamber_for_cells(cur_img, cur_scaled_img, cur_pred, img_name, ac_threshold=ac_threshold)
        cell_dict[img_name] = cells

    # output cells
    outfile = os.path.join(folder, 'threshold_preds_{}.txt'.format(ac_threshold))
    with open(outfile, 'w') as fout:
        json.dump(cell_dict, fout)
    fout.close()
    return cell_dict


if __name__ == '__main__':
    threshold_chambers(folder=img_folder, ac_threshold=0.5)     # allow larger ac chamber
    # threshold_chambers(folder=img_folder, ac_threshold=0.95)    # conservative
