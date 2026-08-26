import numpy as np
import random, os, subprocess, cv2, json
from sys import platform
import time
from matplotlib import pyplot as plt
from matplotlib import patches
from make_accell_data import in_ac_chamber

# img folders
from sys import platform
if platform == "linux" or platform == "linux2":
    plt.switch_backend('agg')
    # linux
    prefix = '/data/yue/pepple/'

    segmentation_json_folder = '/home/ayl/data/pepple/accell/jsons'
    # img_folder = '/home/ayl/data/pepple/accell/segmentations'
    img_folder = '/data/yue/pepple/accell/segmentations'
    empty_img_folder = '/home/ayl/data/pepple/accell/empty_segmentations'

    npy_data_folder = '/home/yue/pepple/accell/npy_data'

    # includes more (unique) images
    chamber_weights_folder = './runs/2017-12-12-17-00-53'
elif platform == "win32":
    # Windows...
    prefix = 'z:/yue/pepple/'

    segmentation_json_folder = './accell/jsons'
    img_folder = os.path.join(prefix, 'accell/segmentations')
    empty_img_folder = './accell/empty_segmentations'

    npy_data_folder = './accell/npy_data'
    chamber_weights_folder = './runs/runAllUnique'

test_weights = 'weights-improvement-174--0.83855137.hdf5'

from predict_image_cells import get_mask_boundary
from make_accell_data import get_img_predictions, visualize_img_prediction, avg_images, overlap, get_patch_stats, \
    calc_cell_class, calc_cell_class_thresh, in_ac_chamber, find_chamber_center, make_box_coords, _box_area, \
    make_overlapping_box, DOWNSAMPLE_RATIO, ACCELL_DIAMETER, RAW_IMG_COLS, RAW_IMG_ROWS
from analyseCellPreds import get_true_coords, get_true_coords_by_class


patch_rows = 32
patch_cols = 32


def visualise_img_patch(img, patch_coords, fig_num=1):
    plt.figure(fig_num)
    plt.clf()
    plt.imshow(img)
    if isinstance(patch_coords, np.ndarray):
        plt.scatter(x=patch_coords[:, 1], y=patch_coords[:, 0], c='yellow', s=1)  # show coords
    else:
        if len(patch_coords)==4:
            x_start, y_start, x_end, y_end = patch_coords
            plt.axes().add_patch(patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, fill=False,
                                                   color='green'))  # show box
        elif len(patch_coords)==2:
            x, y = patch_coords
            plt.scatter(x=[x], y=[y], c='yellow', s=1)  # show coords
        else:
            print('invalid number of coords', patch_coords)
    return


def compute_overlap(cell_coords, target_coords):
    overlap_coords = make_overlapping_box(cell_coords, target_coords)
    cell_area = _box_area(cell_coords)
    target_area = _box_area(target_coords)
    overlap_area = _box_area(overlap_coords)    # intersection area
    union_area = cell_area + target_area - overlap_area
    return overlap_area/union_area, overlap_coords


def make_patches(img_names, convert_imgs_orig_scale, img_preds, out_folder, scaled_imgs=None, patch_overlap=0,
                 labelled_dict=None, visualise=False):
    # for each image and its predicted chamber
    # create smallest 32*32 patches cover of predicted chamber for predicting ac cells
    half_col = int(patch_cols / 2)
    half_row = int(patch_rows / 2)
    for idx, img_name in enumerate(img_names):
        # raw_image = raw_images[idx, ]
        raw_image = convert_imgs_orig_scale[idx, ]   # need to predict accells on converted png images
        raw_rows, raw_cols = raw_image.shape        # img_base = img_name.replace('.TIFF', '')
        img_base = img_name.split('.')[0]
        # img_num = int(img_base.split('(')[1].split(')')[0])
        img_num = img_base.split('_')[-1]   # better formatted in aarons pngs

        cur_pred = img_preds[idx, ]
        chamber_limits, x1, y1, x2, y2 = get_mask_boundary(cur_pred)
        visualize_img_prediction(raw_image, scaled_imgs[idx, ], cur_pred, chamber_limits)

        raw_img_mean, raw_img_std, raw_img_shape = get_patch_stats(raw_image)

        # create patches based on mask on raw image (*DOWNSAMPLE_RATIO)
        # min=0, max=image_edge or chamber_edge + some margin
        x_start = max(int(x1*DOWNSAMPLE_RATIO) - half_col, 0)
        x_end = min(int(x2*DOWNSAMPLE_RATIO) + patch_cols, raw_cols-patch_cols)
        y_start = max(int(y1*DOWNSAMPLE_RATIO) - half_row, 0)
        y_end = min(int(y2*DOWNSAMPLE_RATIO) + patch_rows, raw_rows-patch_rows)
        if visualise:
            print(get_patch_stats(convert_imgs_orig_scale[:, y_start:y_end, x_start:x_end]))
            visualise_img_patch(raw_image, [x_start, y_start, x_end, y_end])

        if labelled_dict is not None:
            img_cells = labelled_dict[img_name.replace('.png', '')]
        else:
            img_cells = []

        for x in range(x_start, x_end, patch_cols-patch_overlap):
            for y in range(y_start, y_end, patch_rows-patch_overlap):
                cur_image = raw_image[y:y+patch_rows, x:x+patch_cols]  # careful about meaning of y(row) and x(col)
                cur_path = os.path.join(out_folder, '{}_h{}_w{}.png'.format(img_base, y, x))
                # # check for cells in this patch and write to file

                cv2.imwrite(cur_path, cur_image)
                if cur_image.shape != (patch_rows, patch_cols):
                    print('accell img_patch creation dimension mismatch -', cur_path, cur_image.shape)

                if visualise:
                    visualise_img_patch(raw_image, [x_start, y_start, x_end, y_end])
                    plt.scatter(x=[x1*DOWNSAMPLE_RATIO, x1*DOWNSAMPLE_RATIO, x2*DOWNSAMPLE_RATIO, x2*DOWNSAMPLE_RATIO],
                                y=[y1*DOWNSAMPLE_RATIO, y2*DOWNSAMPLE_RATIO, y1*DOWNSAMPLE_RATIO, y2*DOWNSAMPLE_RATIO],
                                c='red', s=2)
                    # plt.scatter(x=[x, x, x+patch_cols, x+patch_cols], y=[y, y+patch_rows, y, y+patch_rows], c='lime', s=2)
                    plt.axes().add_patch(patches.Rectangle((x, y), patch_cols, patch_rows, fill=False, color='red'))  # show box
                print(img_name, x, y)
    return 1


def make_ac_test_data(folder=img_folder, do_avg=False, patch_overlap=0):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)
    labelled_cell_jsons = os.path.join('accell', 'jsons_recentered_1scan')
    if do_avg:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered')
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)
        print('mu_raw={}, std_raw={}; mu_avg={}, std_avg={};'.format(np.mean(raw_images_old), np.std(raw_images_old), np.mean(raw_images), np.std(raw_images)))
        k = np.random.randint(0, len(raw_images), 1)[0]    # visualise random raw vs averaged
        plt.figure(1)
        plt.clf()
        plt.subplot(131)
        plt.imshow(raw_images[k,])
        plt.title('averaged {}'.format(img_names[k]))
        plt.subplot(132)
        plt.imshow(raw_images_old[k,])
        plt.title('raw {}'.format(img_names[k]))
        plt.subplot(133)
        temp = raw_images[k,]-raw_images_old[k,]
        print('diff_q={}'.format(np.percentile(temp, q=[0, 5, 10, 50, 90, 95, 100])))
        plt.imshow(temp)
        plt.title('diff {}'.format(img_names[k]))
    true_dict = get_true_coords(labelled_cell_jsons)    # labelled dict - already recentered
    # # faster-rcnn input format
    # true_dict_by_class = get_true_coords_by_class(true_file_path, coord_file=coord_file, path_prefix=path_prefix)

    out_folder = os.path.join(prefix, 'pepple_test_data2{}{}').format('_avg' if do_avg else '', '_overlap' if patch_overlap else '')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # last_idx = 30   # end of valid
    valid_indices = [idx for idx, x in enumerate(img_names) if 'Kathryn' not in x and 'Leslie' not in x]
    valid_img_names = [img_names[x] for x in valid_indices]

    make_patches(valid_img_names, raw_images[valid_indices,], img_preds[valid_indices, ], out_folder,
                 scaled_imgs=converted_imgs[valid_indices,], patch_overlap=patch_overlap, labelled_dict=true_dict, visualise=False)
    return


# for human experts to segment/count accells
def create_avg_test_images(folder=img_folder, do_avg=True):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)

    out_folder = os.path.join('pepple_test_images_avg')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # last_idx = 30   # end of valid
    valid_indices = [idx for idx, x in enumerate(img_names) if 'Kathryn' not in x and 'Leslie' not in x]
    valid_img_names = []
    test_images = []
    for idx in valid_indices:
        cur_img_name = img_names[idx]
        valid_img_names.append(cur_img_name)
        cur_img = raw_images[idx,]
        test_images.append(cur_img )
        # write file
        cv2.imwrite(os.path.join(out_folder, cur_img_name), cur_img)
    test_images = np.asarray(test_images)
    return test_images, valid_indices, valid_img_names


# check whole image patches
def check_ac_test_data(folder):
    if platform == "win32":
        folder = os.path.join('z:/yue/pepple/', folder)

    img_files = [x for x in os.listdir(folder) if '.png' in x]
    img_data = []
    for idx, img_file in enumerate(img_files):
        cur_img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_GRAYSCALE)
        img_data.append(cur_img)
    img_data = np.array(img_data)

    from make_accell_data import get_patch_stats
    print(folder, get_patch_stats(img_data))
    return


def create_test_data_annotation_file(do_avg=True, patch_overlap=0, visualise=False):
    out_folder = os.path.join(prefix, 'pepple_test_data').format('_avg' if do_avg else '', '_overlap' if patch_overlap else '')
    # out_folder = os.path.join(prefix, 'pepple_test_data2{}{}').format('_avg' if do_avg else '', '_overlap' if patch_overlap else '')
    img_names = [x.replace('.png', '') for x in sorted(os.listdir(out_folder)) if '.png' in x]
    labelled_cell_jsons = os.path.join('accell', 'jsons_recentered_1scan')
    if do_avg:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered')
    true_dict = get_true_coords(labelled_cell_jsons)    # labelled dict - already recentered
    # # faster-rcnn input format
    # true_dict_by_class = get_true_coords_by_class(true_file_path, coord_file=coord_file, path_prefix=path_prefix)

    labelled_cells_for_patches_file = os.path.join(out_folder, 'labelled_cells.txt')
    labelled_cells_for_patches_class_file = os.path.join(out_folder, 'labelled_cells_class.txt')
    for idx, img_name in enumerate(img_names):  # for each image patch
        cur_path = os.path.join(out_folder, '{}.png'.format(img_name))
        img_toks = img_name.split('_')
        base_img_name = '_'.join(img_toks[:-2])
        h, w = int(img_toks[-2].replace('h', '')), int(img_toks[-1].replace('w', ''))   # base coord
        x1_p, y1_p, x2_p, y2_p = w, h, w+patch_cols, h+patch_rows   # patch coords
        coord_2 = x1_p, y1_p, x2_p, y2_p
        img_cells = true_dict[base_img_name]

        patch_has_cell = False
        for cell in img_cells:  # for all labelled cells
            coord_1 = make_box_coords(cell, (RAW_IMG_ROWS, RAW_IMG_COLS), box_size=ACCELL_DIAMETER)
            # cell_overlaps_patch = overlap(coord_1, coord_2)
            iou_ratio, overlap_coords = compute_overlap(coord_1, coord_2)   # does cell overlap with patch?
            if iou_ratio>(0.5*ACCELL_DIAMETER**2/(patch_rows*patch_cols)):
                cur_image = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
                patch_has_cell = True
                x1_c, y1_c, x2_c, y2_c = overlap_coords
                x1_c_adj, y1_c_adj, x2_c_adj, y2_c_adj = x1_c-w, y1_c-h, x2_c-w, y2_c-h
                coord_adj = x1_c_adj, y1_c_adj, x2_c_adj, y2_c_adj

                if visualise:
                    plt.imshow(cur_image)
                    plt.scatter(x=[x1_c_adj, x2_c_adj], y=[y1_c_adj, y2_c_adj], c='red', s=2)

                # cell_type = calc_cell_class(coord_adj , cur_image)
                cell_type = calc_cell_class_thresh(cur_image[y1_c_adj:y2_c_adj, x1_c_adj:x2_c_adj], thresh_val=27)
                vals = [cur_path] + [str(x) for x in coord_adj] + ['cell']
                vals_class = [cur_path] + [str(x) for x in coord_adj] + [cell_type]

                with open(labelled_cells_for_patches_file, 'a') as fin:
                    fin.write('{}\n'.format(','.join(vals)))
                fin.close()
                with open(labelled_cells_for_patches_class_file, 'a') as fin:
                    fin.write('{}\n'.format(','.join(vals_class)))
                fin.close()

        if not patch_has_cell:  # write blanks
            vals = [cur_path] + ['']*5
            vals_class = vals
            with open(labelled_cells_for_patches_file, 'a') as fin:
                fin.write('{}\n'.format(','.join(vals)))
            fin.close()
            with open(labelled_cells_for_patches_class_file, 'a') as fin:
                fin.write('{}\n'.format(','.join(vals_class)))
            fin.close()
    return


def make_annotation_levels(patch_folder=os.path.join(prefix, 'pepple_test_data'), annotation_file='labelled_cells.txt',
                           orig_img_folder=os.path.join(prefix, 'accell', 'segmentations'), level=1.0, visualise=False):
    annotation_path = os.path.join(patch_folder, annotation_file)
    # out_file = os.path.join(patch_folder, 'labelled_cells_{}.txt'.format(level))
    out_file = os.path.join(patch_folder, 'labelled_cells_{}_empty.txt'.format(level))
    img_dict = {}
    cell_count = 0
    with open(annotation_path, 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            patch_name = l_toks[0]
            if np.all(np.array(l_toks[1:])!=''):
                x1, y1, x2, y2, cell_type = l_toks[1:]
                patch_path = os.path.join(patch_folder, patch_name)
                patch_img = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                cell = patch_img[int(y1):int(y2), int(x1):int(x2)]  # NB y is row and x column
                if visualise:
                    plt.figure(1)
                    plt.imshow(patch_img)
                    plt.scatter(x=[int(x1), int(x2)], y=[int(y1), int(y2)], c='red')
                cell_mean = np.mean(cell)

                name_toks = patch_name.split('_')
                img_name = '_'.join(name_toks[:-2])
                if img_name in img_dict:
                    img_mean = img_dict[img_name]
                else:
                    img_path = os.path.join(orig_img_folder, '{}.png'.format(img_name))
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_mean = np.mean(img)
                    img_dict[img_name] = img_mean

                brightness_ratio = cell_mean/img_mean
                if brightness_ratio>level:
                    cell_count += 1
                    with open(out_file, 'a') as fout:
                        val_data = [patch_name] + l_toks[1:]
                        fout.write('{}\n'.format(','.join(val_data)))
                    fout.close()
                else:   # ignore - not bright enough
                    # print('{} not a cell at level={}; brightness={}, cell_mean={}, img_mean={}'.format(l_toks, level, brightness_ratio, cell_mean, img_mean))
                    with open(out_file, 'a') as fout:
                        val_data = [patch_name] + ['']*5
                        fout.write('{}\n'.format(','.join(val_data)))
                    fout.close()
            else:   # ignore - no cells
                with open(out_file, 'a') as fout:
                    val_data = [patch_name] + [''] * 5
                    fout.write('{}\n'.format(','.join(val_data)))
                fout.close()
    fin.close()
    print('{} cells at level={}'.format(cell_count, level))
    return


if __name__ == '__main__':
    # # predict all inflamed ac chambers
    make_ac_test_data(folder=img_folder, do_avg=True)
    # create_test_data_annotation_file(do_avg=True)
    create_test_data_annotation_file(do_avg=False)
    # make_annotation_levels(patch_folder=os.path.join(prefix, 'pepple_test_data'), annotation_file='labelled_cells.txt',
    #                        orig_img_folder=os.path.join(prefix, 'accell', 'segmentations'), level=1.0)
    # make_annotation_levels(patch_folder=os.path.join(prefix, 'pepple_test_data'), annotation_file='labelled_cells.txt',
    #                        orig_img_folder=os.path.join(prefix, 'accell', 'segmentations'), level=1.25)
    # make_annotation_levels(patch_folder=os.path.join(prefix, 'pepple_test_data'), annotation_file='labelled_cells.txt',
    #                        orig_img_folder=os.path.join(prefix, 'accell', 'segmentations'), level=1.5)
    # make_annotation_levels(patch_folder=os.path.join(prefix, 'pepple_test_data'), annotation_file='labelled_cells.txt',
    #                        orig_img_folder=os.path.join(prefix, 'accell', 'segmentations'), level=1.75)
    make_annotation_levels(patch_folder=os.path.join(prefix, 'pepple_test_data'), annotation_file='labelled_cells.txt',
                           orig_img_folder=os.path.join(prefix, 'accell', 'segmentations'), level=2.0)
    make_annotation_levels(patch_folder=os.path.join(prefix, 'pepple_test_data'), annotation_file='labelled_cells.txt',
                           orig_img_folder=os.path.join(prefix, 'accell', 'segmentations'), level=2.25)
    make_annotation_levels(patch_folder=os.path.join(prefix, 'pepple_test_data'), annotation_file='labelled_cells.txt',
                           orig_img_folder=os.path.join(prefix, 'accell', 'segmentations'), level=2.5)
    1

    # TODO need to raw images (1024*1000) converted (512*500) with cv2.resize instead of imagemagick
    # raw files same place (also need avg'd files for easy comparison)
    # masks still available in /data/yue/pepple_dqn/acseg/segmentations
    # also get empty_segmentations (ask aaron if not found)
    # create patches 384*128 or something
    # store and retrieve as npy for speed?

    # make_ac_test_data(folder=img_folder, do_avg=True, overlap=16)
    #
    # ## allow overlap
    # # make_ac_test_data(folder=img_folder, do_avg=True, overlap=16)
    #
    # # create_avg_test_images(folder=img_folder, do_avg=True)
    #
    # # check patches generated
    # check_ac_test_data(os.path.join('pepple_test_data'))
    # check_ac_test_data(os.path.join('pepple_test_data_avg'))
    # check_ac_test_data(os.path.join('pepple_test_data_avgNew'))
    # check_ac_test_data(os.path.join('pepple_test_data_avg_overlap'))
    # check_ac_test_data(os.path.join('pepple_test_data2_avg'))