import numpy as np
import os, json
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

from make_accell_data import get_img_predictions, avg_images, get_patch_stats, make_box_coords, calc_cell_class, \
    DOWNSAMPLE_RATIO, ACCELL_DIAMETER, RAW_IMG_COLS
from analyseCellPreds import visualise_pred_vs_truth, check_missed_helper, \
    check_false_positives_helper, get_true_coords, visualise_preds_minus_chamber, plot_img_boxes, recombine_predictions
from predict_image_cells import get_predicted_coords, is_cell_in_ac, calc_img_chamber_size, any_middle_stripes

class_colors = {'cell': 'red', 'cell_medium': 'pink', 'cell_lite': 'white', 'all': 'white'}
# class_colors_rgb = {'cell': (255, 0, 0), 'cell_medium': (255, 192, 203), 'cell_lite': (255, 255, 255), 'all': (255, 0, 0)}
class_colors2 = {'cell': (0, 0, 255), 'cell_medium': (203, 192, 255), 'cell_lite': (255, 255, 255), 'all': (0, 0, 255)}


def parse_predictions(results_folder, classes=['cell', 'cell_medium', 'cell_lite'], suffix='', true_dict=None):
    all_dict = {}
    for idx, cls in enumerate(classes):
        # class_pred_dict = get_predicted_coords(os.path.join(results_folder, 'coords_{}.txt'.format(cls)))
        class_pred_dict = get_predicted_coords(results_folder, 'coords_{}_{}.txt'.format(cls, suffix))
        class_combined_pred_dict = recombine_predictions(class_pred_dict)
        all_dict[cls] = class_combined_pred_dict
        if true_dict:
            visualise_pred_vs_truth(true_dict, class_combined_pred_dict, class_type='cell')
    return all_dict


# reads prediction files, vs human labels, visualise on images and patches
def whole_img_sensitivity(img_folder, test_folder='pepple_test_data', suffix='weights_s128', ac_threshold=.3, pixel_lim=3):
    do_avg = True if 'avg' in test_folder else False
    if do_avg:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered')
    else:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered_1scan')

    # for visualisation and checking preds actually in chamber
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder=img_folder)
    # dont need avg to visualise or get chamber coords! so skip do_avg for speed!
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)

    true_dict = get_true_coords(labelled_cell_jsons)
    valid_img_names = get_valid_images(img_names)
    true_dict = get_true_dict_for_validation(true_dict, valid_img_names)
    # removes duplicate/neighbouring cells
    true_dict, duplicate_cells_dict = check_neighbouring_cells_from_dict(true_dict)
    # constrain by boundary just like DL accell for direct comparison without worrying about boundary
    true_dict_wrapper = constrain_by_ac_chamber({'all':true_dict}, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)
    true_dict = true_dict_wrapper['all']    # alias
    # break up by class - think about
    true_dict_by_class = get_true_coords_by_class(true_dict, imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    all_dict = parse_predictions(test_folder, suffix=suffix)  # {cell:{img_name:[coords]}, cell_medium:{}}

    output_dir = os.path.join(test_folder, suffix)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # check if these preds in chamber given preds
    valid_json_file = os.path.join(output_dir, 'valid_coords_{}.json'.format(suffix))
    if not os.path.isfile(valid_json_file):
        all_valid_dict = {}
        for cls, cls_dict in all_dict.items():
            all_valid_dict[cls] = {}
            for img_name, img_coords in cls_dict.items():
                temp = get_valid_coords(img_name, img_coords, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)  # need data to process for each image
                all_valid_dict[cls][img_name] = temp

        # store for future reference/load
        with open(valid_json_file, 'w') as fout:
            json.dump(all_valid_dict, fout)
        fout.close()
    else:
        fin = open(valid_json_file).read()
        all_valid_dict = json.loads(fin)

    # combine for overall value
    combined_dict = combine_class_predictions(all_dict)     # doesnt care about chamber
    # removes duplicate/neighbouring cells
    combined_dict, duplicate_combined_dict = check_neighbouring_cells_from_dict(combined_dict)
    combined_valid_dict = combine_class_predictions(all_valid_dict)
    combined_valid_dict_no_lite = combine_class_predictions(all_valid_dict, ignore_classes=['cell_lite'])

    # sensitivity
    log_sensitivity_precision(output_dir, true_dict, combined_dict, class_type='total', pixel_lim=pixel_lim)
    log_sensitivity_precision(output_dir, true_dict, combined_valid_dict, class_type='total_valid', pixel_lim=pixel_lim)
    log_sensitivity_precision(output_dir, true_dict, combined_valid_dict, class_type='total_valid_no_lite', pixel_lim=pixel_lim)
    # classes
    classes = all_dict.keys()
    for cls in classes:
        log_sensitivity_precision(output_dir, true_dict_by_class[cls], all_valid_dict[cls], class_type=cls, pixel_lim=pixel_lim)

    # save images - raw, ac_chamber, predicted_cells
    save_imgs(output_dir, true_dict, all_valid_dict, img_names, raw_images, img_preds, ac_threshold=ac_threshold)
    return


def constrain_by_ac_chamber(all_dict, img_names, img_preds, raw_images, converted_imgs, ac_threshold=.9):
    all_valid_dict = {}
    for cls, cls_dict in all_dict.items():
        all_valid_dict[cls] = {}
        for img_name, img_coords in cls_dict.items():
            temp = get_valid_coords(img_name, img_coords, img_names, img_preds, raw_images, converted_imgs,
                                    ac_threshold=ac_threshold)  # need data to process for each image
            all_valid_dict[cls][img_name] = temp
    return all_valid_dict


def save_imgs(output_dir, true_dict, pred_dict_by_class, img_names, imgs, img_preds, ac_threshold=.3, ignore_classes=['cell_lite']):
    classes = list(pred_dict_by_class.keys())
    # classes = ['cell', 'cell_medium', 'cell_lite']
    # classes = ['all']
    if len(ignore_classes)>0:
        classes = list(set(classes) - set(ignore_classes))

    for fname, f_coords in true_dict.items():
        f_idx = img_names.index('{}.png'.format(fname))
        cur_img_ac_pred = img_preds[f_idx,]
        cur_raw_img = imgs[f_idx, ].astype(np.float32)
        # cur_raw_img = cv2.cvtColor(cur_raw_img.copy(), cv2.COLOR_GRAY2BGR)
        chamber_limits, chamber_size, mean_x, mean_y = calc_img_chamber_size(cur_img_ac_pred,
                                                                             pred_threshold=ac_threshold)
        cur_mask = make_mask_from_preds(cur_img_ac_pred, chamber_limits)

        # plt.clf()
        # plt.subplot(131)
        # plt.imshow(cur_raw_img)
        # plt.title(fname)
        # plt.subplot(132)
        # plt.imshow(cur_raw_img[::2, ::2])   # every other pixel downsampling for visual purposes
        # plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='yellow', s=1)    # NB. y in col 0
        # plt.scatter(x=mean_x, y=mean_y, c='green', s=2)
        # plt.title('with seg_chamber')
        #
        # plt.subplot(133)
        # plt.imshow(cur_raw_img)
        # plt.title('with preds')
        # for img_coord in f_coords:  # true coords (x,y)
        #     plt.scatter(x=round(img_coord[0]), y=round(img_coord[1]), c='yellow', s=2)
        # for cls in classes: # pred coords color-coded by class
        #     class_preds_dict = pred_dict_by_class[cls]
        #     cls_color = class_colors[cls]
        #     if fname in class_preds_dict:
        #         f_pred_coords = class_preds_dict[fname]
        #         for img_coord in f_pred_coords:  # (x1, y1, x2, y2)
        #             if len(img_coord)==4:
        #                 x1, y1, x2, y2 = img_coord
        #             elif len(img_coord)==5:
        #                 x1, y1, x2, y2, prob = img_coord
        #             elif len(img_coord)==2:
        #                 x1, y1 = img_coord
        #                 x2, y2 = x1, y1
        #             x = round(x1 + x2)/2
        #             y = round(y1 + y2) / 2
        #             plt.scatter(x=x, y=y, c=cls_color,s=2)
        # plt.savefig(os.path.join(output_dir, '{}.png'.format(fname)), bbox_inches='tight')

        # annotated img
        cur_raw_img_color = np.repeat(np.expand_dims(cur_raw_img, axis=2), 3, axis=2)
        annotated_img = add_cells_to_img(fname, cur_raw_img_color, f_coords, pred_dict_by_class, classes=classes)
        raw_plus_accell = cv2.hconcat((cur_raw_img_color, annotated_img))
        cv2.imwrite(os.path.join(output_dir, '{}_accell.png'.format(fname)), raw_plus_accell)
        # ac_seg img
        raw_plus_acseg = cv2.hconcat(( cur_raw_img[::2,::2], cur_mask[:, :int(RAW_IMG_COLS/DOWNSAMPLE_RATIO)] ))
        cv2.imwrite(os.path.join(output_dir, '{}_acseg.png'.format(fname)), raw_plus_acseg)
    return


def make_mask_from_preds(cur_pred, chamber_limits, visualise=False):
    cur_mask = np.zeros(cur_pred.shape, dtype=np.float32)
    cur_mask[tuple(np.transpose(chamber_limits))] = 255
    if visualise:
        plt.imshow(cur_mask)
    return cur_mask


def add_cells_to_img(fname, img, f_coords, pred_dict_by_class, classes=['cell', 'cell_medium', 'cell_lite', 'not_cell', 'not_cell_medium', 'not_cell_lite'], visualise=False):
    img_cp = img.copy()
    for img_coord in f_coords:  # true coords (x,y)
        cv2.circle(img_cp, tuple(img_coord), radius=1, color=(0, 255, 255), thickness=2)  # yellow

    for cls in classes:  # pred coords color-coded by class
        class_preds_dict = pred_dict_by_class[cls]
        cls_color = class_colors2[cls]
        if fname in class_preds_dict:
            f_pred_coords = class_preds_dict[fname]
            for img_coord in f_pred_coords:  # (x1, y1, x2, y2)
                if len(img_coord) == 4:
                    x1, y1, x2, y2 = img_coord
                elif len(img_coord) == 5:
                    x1, y1, x2, y2, prob = img_coord
                elif len(img_coord) == 2:
                    x1, y1 = img_coord
                    x2, y2 = x1, y1
                cv2.rectangle(img_cp, (x1, y1), (x2, y2), color=cls_color, thickness=2)
    if visualise:
        plt.figure(100)
        plt.imshow(img.astype(np.int))
        plt.figure(101)
        plt.imshow(img_cp.astype(np.int))
    return img_cp


def log_sensitivity_precision(output_dir, true_dict, pred_dict, class_type='total', pixel_lim=3):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'summary_stats.txt')

    found_dict, missed_dict = check_missed_helper(true_dict, pred_dict, pixel_lim=pixel_lim)
    total_labelled = np.sum([len(val) for val in true_dict.values()])
    total_found = np.sum([len(val) for val in found_dict.values()])
    total_missed = np.sum([len(val) for val in missed_dict.values()])
    sensitivity = total_found / total_labelled

    false_positive_dict, matched_dict = check_false_positives_helper(pred_dict, true_dict, pixel_lim=pixel_lim)
    total_predicted = np.sum([len(val) for val in pred_dict.values()])
    total_fp = np.sum([len(val) for val in false_positive_dict.values()])
    total_matched = np.sum([len(val) for val in matched_dict.values()])
    fpr = total_fp/total_predicted
    precision = 1-fpr
    fscore = 2*(sensitivity*precision)/(sensitivity+precision)

    write_type = 'w' if 'total' in class_type else 'a'
    with open(output_file, write_type) as fout:
        vals =['type', 'labelled', 'found', 'missed', 'sensitivity', 'predicted', 'matched', 'fp', 'fpr', 'precision', 'fscore']
        fout.write('{}\n'.format(','.join(vals)))
        vals2 = [class_type, total_labelled, total_found, total_missed, sensitivity, total_predicted,
                   total_matched, total_fp, fpr, precision, fscore]
        fout.write('{},{}\n'.format(vals2[0], ','.join(['{:0.2f}'.format(x) for x in vals2[1:]])))
    fout.close()

    # individual files data
    with open(os.path.join(output_dir, 'found_{}.json'.format(class_type)), 'w') as fout:
        json.dump(found_dict, fout)
    fout.close()
    with open(os.path.join(output_dir, 'missed_{}.json'.format(class_type)), 'w') as fout:
        json.dump(missed_dict, fout)
    fout.close()
    with open(os.path.join(output_dir, 'fp_{}.json'.format(class_type)), 'w') as fout:
        json.dump(false_positive_dict, fout)
    fout.close()
    with open(os.path.join(output_dir, 'matched_{}.json'.format(class_type)), 'w') as fout:
        json.dump(matched_dict, fout)
    fout.close()
    return


def combine_class_predictions(all_dict, ignore_classes=['cell_lite']):
    classes = all_dict.keys()
    all_img_names = []  # img_names with prediction
    combined_dict = {}
    for cls in classes:
        if cls not in ignore_classes:
            all_img_names += all_dict[cls].keys()  # imgs predicted in each cell class
    all_img_names = list(set(all_img_names))  # unique keys

    for img_name in all_img_names:
        combined_dict[img_name] = []  # init
        for cls in classes:
            if cls not in ignore_classes:
                combined_dict[img_name] += all_dict[cls][img_name] if img_name in all_dict[cls] else []
    return combined_dict


def get_valid_coords(fname, img_coords, img_names, img_preds, raw_images, converted_imgs, ac_threshold=.3, visualise=False):
    valid_coords = []
    f_idx = img_names.index('{}.png'.format(fname))
    cur_img_ac_pred = img_preds[f_idx, ]
    cur_raw_img = raw_images[f_idx, ]
    cur_converted_img = converted_imgs[f_idx, ]

    # low ac_threshold means bigger chambers -> potentially more false positives no in chamber
    # alternative is missing cells in chamber if chamber predictions are inaccurate
    chamber_limits, chamber_size, mean_x, mean_y = calc_img_chamber_size(cur_img_ac_pred , pred_threshold=ac_threshold)
    mid_limits, mid_min, mid_max = any_middle_stripes(cur_raw_img, avg_period=10, intensity_threshold=180)  # should do this on correct (avg or not img)

    for img_coord in img_coords:
        coord_in_ac = is_cell_in_ac(img_coord, chamber_limits, mean_x, mean_y, conservative=0, no_mid=1, 
                                    mid_min=mid_min, mid_max=mid_max, img=cur_raw_img, visualise=False)
        if coord_in_ac:
            valid_coords.append(img_coord)

    if len(valid_coords)!=len(img_coords) or visualise:
        plt.figure(0)
        plt.imshow(cur_raw_img[::2, ::2])   # every other pixel downsampling for visual purposes
        plt.figure(1)
        plt.clf()
        plt.imshow(cur_raw_img[::2, ::2])   # every other pixel downsampling for visual purposes
        plt.title(fname)
        plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='yellow', s=1)    # NB. y in col 0
        plt.scatter(x=mean_x, y=mean_y, c='green', s=2)
        for img_coord in img_coords:
            plt.scatter(x=round(img_coord[0]/DOWNSAMPLE_RATIO), y=round(img_coord[1]/DOWNSAMPLE_RATIO), c='blue', s=2)
        for valid_coord in valid_coords:
            plt.scatter(x=round(valid_coord[0]/DOWNSAMPLE_RATIO), y=round(valid_coord[1]/DOWNSAMPLE_RATIO), c='red',s=2)
        1
    return valid_coords


def get_true_coords_by_class(true_dict, imgs, img_names, visualise=False):
    true_dict_by_class = {}

    for fname, img_coords in true_dict.items():
        f_idx = img_names.index('{}.png'.format(fname))
        cur_img = imgs[f_idx, ]
        img_shape = cur_img.shape
        for img_coord in img_coords:    # (x,y)
            cell_coords = make_box_coords(img_coord, img_shape, box_size=ACCELL_DIAMETER)  # (x,y) -> (x1, y1, x2, y2)
            x1, y1, x2, y2 = cell_coords
            cell = cur_img[y1:y2, x1:x2]
            cell_type, cell_brightness = calc_cell_class(cell, cur_img)  # based on intensity
        
            if visualise:
                print(img_coord, np.mean(cur_img), np.mean(cell), cell_type)
                # plt.figure(1)
                # plt.clf()
                # plt.imshow(cur_img)
                # plt.scatter(x=img_coord[0], y=img_coord[1], c='red', s=1)
                # plt.axes().add_patch(patches.Rectangle((x1, y1), ACCELL_DIAMETER, ACCELL_DIAMETER, fill=False, color='white'))  # show box
                plot_img_boxes(fname, cur_img, cell_coords, color='white')
        
            if cell_type not in true_dict_by_class:
                true_dict_by_class[cell_type] = {}
                true_dict_by_class[cell_type][fname] = [img_coord]
            else:
                if fname not in true_dict_by_class[cell_type]:
                    true_dict_by_class[cell_type][fname] = [img_coord]
                else:
                    true_dict_by_class[cell_type][fname].append(img_coord)
    return true_dict_by_class


def get_valid_images(img_names):
    valid_img_names = []
    for idx, img_name in enumerate(img_names):
        if 'Kathryn' not in img_name and 'Leslie' not in img_name:
            valid_img_names.append(img_name)
    return valid_img_names


def get_true_dict_for_validation(true_dict, valid_img_names):
    true_dict_for_validation = {}
    for fname, f_data in true_dict.items():
        if '{}.png'.format(fname) in valid_img_names:
            true_dict_for_validation[fname] = f_data
    return true_dict_for_validation


def distance(x, ys, axis=1):
    d = np.linalg.norm(x - ys, axis=axis)
    return d


def check_neighbouring_cells(json_folder):
    labelled_cells_dict = get_true_coords(json_folder)
    unique_cells_dict, duplicate_cells_dict = check_neighbouring_cells_from_dict(labelled_cells_dict)
    return unique_cells_dict, duplicate_cells_dict


def check_neighbouring_cells_from_dict(labelled_cells_dict, cell_diam=ACCELL_DIAMETER):
    unique_cells_dict = {}
    duplicate_cells_dict = {}
    for fname, f_coords in labelled_cells_dict.items():
        cur_img_coords = []
        duplicate_coords = []
        num_coords = len(f_coords)
        untreated_coords = np.array(f_coords).copy()

        while len(untreated_coords) > 0:
            cur_coord = untreated_coords[0,]
            cur_img_coords.append(cur_coord)
            other_coords = untreated_coords[1:, ]
            distances = distance(cur_coord, other_coords)

            if len(distances) > 0:
                duplicate_indices = np.nonzero(distances < cell_diam)[0]
            else:
                duplicate_indices = np.array([])

            for idx in duplicate_indices:
                duplicate_coords.append(other_coords[idx,])
                # untreated_coords = np.delete(untreated_coords, idx+1)   # insert
            untreated_coords = np.delete(untreated_coords, np.insert(duplicate_indices + 1, 0, 0),
                                         axis=0)  # insert 0 (cur_coord) to be removed

        unique_cells_dict[fname] = cur_img_coords
        duplicate_cells_dict[fname] = duplicate_coords

    print('all_cells', np.sum([len(vals) for vals in labelled_cells_dict.values()]))
    print('unique_cells', np.sum([len(vals) for vals in unique_cells_dict.values()]))
    print('duplicate_cells', np.sum([len(vals) for vals in duplicate_cells_dict.values()]))
    return unique_cells_dict, duplicate_cells_dict


def make_test_images(img_folder, do_avg=True):
    # for visualisation and checking preds actually in chamber
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder=img_folder)
    # dont need avg to visualise or get chamber coords! so skip do_avg for speed!
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)

    if do_avg:
        out_folder = os.path.join(base_folder, 'averaged_imgs')
    else:
        out_folder = os.path.join(base_folder, '1scan_imgs')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    for idx, img_name in enumerate(img_names):
        cv2.imwrite(os.path.join(out_folder, img_name), raw_images[idx,] )
    return


def review_images(do_avg=True, source='nick'):
    # always recentered folder
    if do_avg:
        folder = os.path.join(base_folder, 'averaged_imgs')
        json_folder = json_folder_recentered
    else:
        folder = os.path.join(base_folder, '1scan_imgs')
        json_folder = json_folder_1scan

    img_names = [x for x in sorted(os.listdir(folder)) if '.png' in x]
    valid_img_names = get_valid_images(img_names)

    make_review_txt_from_json(img_names, valid_img_names, do_avg=do_avg)
    if source=='nick':
        true_dict = get_true_coords(json_folder)
    else:
        review_file = os.path.join(base_folder, 'accell', 'yue_seg_avg.txt')
        true_dict = get_review_txt_dict(review_file=review_file)
    valid_dict = get_true_dict_for_validation(true_dict, valid_img_names)

    for idx, img_name in enumerate(valid_img_names):
        img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.title(img_name)
        
        short_name = img_name.replace('.png', '')
        if short_name in valid_dict:
            img_coords = valid_dict[short_name]
            for coord in img_coords:
                x, y = coord
                x_prime, y_prime = x+ACCELL_DIAMETER, y+ACCELL_DIAMETER
                plt.text(x_prime, y_prime, 'x={},y={}'.format(x,y))
        1
    return


def get_review_txt_dict(review_file, allow_commented=False):
    review_dict = {}
    with open(review_file, 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            l_toks = [x.strip() for x in l_toks]
            num_toks = len(l_toks)
            if num_toks ==3:
                key, x, y = l_toks
                comment = None
            elif num_toks ==4:
                key, x, y, comment = l_toks
            cur_val = [int(x), int(y)]

            # if num_toks==3 or allow_commented:
            if num_toks==3 or comment == 'boundary' or allow_commented:
                if key not in review_dict:
                    review_dict[key] = [cur_val]
                else:
                    review_dict[key].append(cur_val)
    return review_dict


def make_review_txt_from_json(img_names, valid_img_names, do_avg=True):
    out_file = os.path.join(base_folder, 'accell', 'nick_flattened_doAvg={}.txt'.format(do_avg))
    # always recentered folder
    if do_avg:
        folder = json_folder_recentered
    else:
        folder = json_folder_1scan

    true_dict = get_true_coords(folder)
    with open(out_file, 'w') as fout:
        keys = list(true_dict.keys())
        for idx, key in enumerate(sorted(keys)):
            coords = true_dict[key]
            long_key = '{}.png'.format(key)
            if long_key in img_names and long_key in valid_img_names:
                for coord in coords:
                    val = [key] + [str(x) for x in coord]
                    fout.write('{}\n'.format(','.join(val)))
    fout.close()
    return


def compare_nick_vs_yue(img_folder, do_avg=True, ac_threshold=.3, pixel_lim=3):
    if do_avg:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered')
    else:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered_1scan')

    # for visualisation and checking preds actually in chamber
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder=img_folder)
    # dont need avg to visualise or get chamber coords! so skip do_avg for speed!
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)

    true_dict = get_true_coords(labelled_cell_jsons)    # this grabs labels from nick, kathryn and leslie on single scans
    print('all nick labelled', np.sum([len(vals) for vals in true_dict.values()]))
    true_dict, duplicate_cells_dict = check_neighbouring_cells(labelled_cell_jsons)  # removes duplicate/neighbouring cells
    print('all unique nick labelled', np.sum([len(vals) for vals in true_dict.values()]))
    valid_img_names = get_valid_images(img_names)
    true_dict = get_true_dict_for_validation(true_dict, valid_img_names)
    # constrain by boundary just like DL accell for direct comparison without worrying about boundary
    true_dict_wrapper = constrain_by_ac_chamber({'all': true_dict}, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)
    print('all valid nick labelled', np.sum([len(vals) for vals in true_dict.values()]))
    print('all valid constrained labelled', np.sum([len(vals) for vals in true_dict_wrapper['all'].values()]))

    # load yue segmentations
    if do_avg:
        yue_file = os.path.join(base_folder, 'accell', 'yue_seg_avg_rechecked.txt')
    else:
        yue_file = os.path.join(base_folder, 'accell', 'yue_seg_1scan.txt')
    yue_dict = get_review_txt_dict(review_file=yue_file, allow_commented=True)
    print('all yue', np.sum([len(vals) for vals in yue_dict.values()]))
    yue_dict, yue_duplicate_dict = check_neighbouring_cells_from_dict(yue_dict)  # removes duplicate/neighbouring cells
    print('all unique yue', np.sum([len(vals) for vals in yue_dict.values()]))
    yue_dict_wrapper = constrain_by_ac_chamber({'all':yue_dict}, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)
    print('all constrained yue', np.sum([len(vals) for vals in yue_dict_wrapper['all'].values()]))

    # calculate sensitivity and false positive both ways
    # all labelled after accounting for pixel_lim
    output_dir = os.path.join(base_folder, 'nick_vs_yue_ac{}'.format(ac_threshold))
    log_sensitivity_precision(output_dir, true_dict_wrapper['all'], yue_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    # reverse direction of comparison since not commutative
    output_dir = os.path.join(base_folder, 'yue_vs_nick_ac{}'.format(ac_threshold))
    log_sensitivity_precision(output_dir, yue_dict_wrapper['all'], true_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    save_imgs(output_dir, true_dict_wrapper['all'], yue_dict_wrapper, img_names, raw_images, img_preds, ac_threshold=ac_threshold)

    # stricter labels for yue vs nick - take out commented stuff
    yue_dict_strict = get_review_txt_dict(review_file=yue_file, allow_commented=False)
    print('all yue strict', np.sum([len(vals) for vals in yue_dict_strict.values()]))
    yue_dict_strict, yue_duplicate_strict = check_neighbouring_cells_from_dict(yue_dict_strict)  # removes duplicate/neighbouring cells
    print('all unique yue strict', np.sum([len(vals) for vals in yue_dict_strict.values()]))
    yue_dict_strict_wrapper = constrain_by_ac_chamber({'all':yue_dict_strict}, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)
    print('all constrained yue strict', np.sum([len(vals) for vals in yue_dict_strict_wrapper['all'].values()]))

    output_dir = os.path.join(base_folder, 'nick_vs_yue_strict_ac{}'.format(ac_threshold))
    log_sensitivity_precision(output_dir, true_dict_wrapper['all'], yue_dict_strict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    save_imgs(output_dir, true_dict_wrapper['all'], yue_dict_strict_wrapper, img_names, raw_images, img_preds, ac_threshold=ac_threshold)
    output_dir = os.path.join(base_folder, 'yue_vs_nick_strict_ac{}'.format(ac_threshold))
    log_sensitivity_precision(output_dir, yue_dict_strict_wrapper['all'], true_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)

    # break up by class - think about relative intensity differences
    true_dict_by_class = get_true_coords_by_class(true_dict_wrapper['all'], imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    # yue_dict_by_class = get_true_coords_by_class(yue_dict_wrapper['all'], imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    yue_dict_strict_by_class = get_true_coords_by_class(yue_dict_strict_wrapper['all'], imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    classes = true_dict_by_class.keys()
    output_dir = os.path.join(base_folder, 'nick_vs_yue_strict2_ac{}'.format(ac_threshold))
    for cls in classes:
        log_sensitivity_precision(output_dir, true_dict_by_class[cls], yue_dict_strict_by_class[cls], class_type=cls, pixel_lim=pixel_lim)
    output_dir = os.path.join(base_folder, 'yue_vs_nick_strict2_ac{}'.format(ac_threshold))
    for cls in classes:
        log_sensitivity_precision(output_dir, yue_dict_strict_by_class[cls], true_dict_by_class[cls], class_type=cls, pixel_lim=pixel_lim)
    return


# load appropriate source into dictionary format
def load_labelled_data(raw_images, converted_imgs, img_names, img_preds, do_avg=True, data_source='nick', ac_threshold=.3):
    if data_source=='nick':
        if do_avg:
            labelled_cell_jsons = os.path.join('accell', 'jsons_recentered')
        else:
            labelled_cell_jsons = os.path.join('accell', 'jsons_recentered_1scan')

        true_dict = get_true_coords(labelled_cell_jsons)
        print('all nick labelled', np.sum([len(vals) for vals in true_dict.values()]))
        true_dict, duplicate_cells_dict = check_neighbouring_cells(labelled_cell_jsons)  # removes duplicate/neighbouring cells
        print('all unique nick labelled', np.sum([len(vals) for vals in true_dict.values()]))
        valid_img_names = get_valid_images(img_names)
        true_dict = get_true_dict_for_validation(true_dict, valid_img_names)
        # constrain by boundary just like DL accell for direct comparison without worrying about boundary
        true_dict_wrapper = constrain_by_ac_chamber({'all': true_dict}, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)
        print('all valid nick labelled', np.sum([len(vals) for vals in true_dict.values()]))
        print('all valid constrained labelled', np.sum([len(vals) for vals in true_dict_wrapper['all'].values()]))
    else:
        if do_avg:
            yue_file = os.path.join(base_folder, 'accell', 'yue_seg_avg_rechecked.txt')
        else:
            yue_file = os.path.join(base_folder, 'accell', 'yue_seg_1scan.txt')
        # yue_dict = get_review_txt_dict(review_file=yue_file, allow_commented=True)
        yue_dict = get_review_txt_dict(review_file=yue_file, allow_commented=False)     # already on validation imgs
        print('all yue', np.sum([len(vals) for vals in yue_dict.values()]))
        yue_dict, yue_duplicate_dict = check_neighbouring_cells_from_dict(yue_dict)  # removes duplicate/neighbouring cells
        print('all unique yue', np.sum([len(vals) for vals in yue_dict.values()]))
        yue_dict_wrapper = constrain_by_ac_chamber({'all': yue_dict}, img_names, img_preds, raw_images, converted_imgs,
                                                   ac_threshold=ac_threshold)
        print('all constrained yue', np.sum([len(vals) for vals in yue_dict_wrapper['all'].values()]))
        true_dict_wrapper = yue_dict_wrapper    # aliasing

    true_dict_by_class = get_true_coords_by_class(true_dict_wrapper['all'], imgs=raw_images, img_names=img_names)
    return true_dict_wrapper, true_dict_by_class 


if __name__ == '__main__':
    from sys import platform
    if platform == "linux" or platform == "linux2":
        base_folder = '/data/yue/pepple'
    elif platform == "win32":
        base_folder = 'z:/yue/pepple'
    orig_img_folder = os.path.join(base_folder, 'accell', 'segmentations')
    empty_img_folder = os.path.join(base_folder, 'accell', 'empty_segmentations')
    json_folder = os.path.join(base_folder, 'accell', 'jsons')
    json_folder_recentered = os.path.join(base_folder, 'accell', 'jsons_recentered')
    json_folder_1scan = os.path.join(base_folder, 'accell', 'jsons_recentered_1scan')

    # review and manually segment cells for images
    # review_images(do_avg=True, source='yue')
    # review_images(do_avg=False, source='yue')

    ## check_neighbouring_cells for duplicacy
    # check_neighbouring_cells(json_folder=json_folder_recentered)
    # make_test_images(orig_img_folder, do_avg=True)
    # make_test_images(orig_img_folder, do_avg=False)

    # # 'CUDA_VISIBLE_DEVICES=3 python /data/yue/keras-frcnn-master/test_frcnn.py -p /data/yue/pepple/pepple_test_data2_avg/ --network vgg --config_filename /data/yue/pepple/accell/ac_training_avg_blurred_32_32/weights_s128/config.pickle'
    whole_img_sensitivity(img_folder=orig_img_folder, test_folder=os.path.join(base_folder, 'pepple_test_data2_avg'), suffix='weights_s128')
    # 'CUDA_VISIBLE_DEVICES=0 python /data/yue/keras-frcnn-master/test_frcnn.py -p /data/yue/pepple/pepple_test_data2_avg/ --network vgg --config_filename /data/yue/pepple/accell/ac_training_avg_insitunickHypo_32_32/weights_s128_2class/config.pickle'
    whole_img_sensitivity(img_folder=orig_img_folder, test_folder=os.path.join(base_folder, 'pepple_test_data2_avg'), suffix='weights_s128_2class')
    # 'CUDA_VISIBLE_DEVICES=1 python /data/yue/keras-frcnn-master/test_frcnn.py -p /data/yue/pepple/pepple_test_data2_avg/ --network vgg --config_filename /data/yue/pepple/accell/ac_training_avg_insitunickHypo_32_32/weights_s128_2class_aug/config.pickle'
    whole_img_sensitivity(img_folder=orig_img_folder, test_folder=os.path.join(base_folder, 'pepple_test_data2_avg'), suffix='weights_s128_2class_aug')
    # 'CUDA_VISIBLE_DEVICES=2 python /data/yue/keras-frcnn-master/test_frcnn.py -p /data/yue/pepple/pepple_test_data2_avg/ --network vgg --config_filename /data/yue/pepple/accell/ac_training_avg_insitunickHypo_32_32/weights_s128_2class_aug_s64/config.pickle'
    whole_img_sensitivity(img_folder=orig_img_folder, test_folder=os.path.join(base_folder, 'pepple_test_data2_avg'), suffix='weights_s128_2class_aug_s64')
    # 'CUDA_VISIBLE_DEVICES=1 python /data/yue/keras-frcnn-master/test_frcnn.py -p /data/yue/pepple/pepple_test_data2_avg/ --network vgg --config_filename /data/yue/pepple/accell/ac_training_avg_blurred_32_32/weights_s64/config.pickle'
    whole_img_sensitivity(img_folder=orig_img_folder, test_folder=os.path.join(base_folder, 'pepple_test_data2_avg'), suffix='weights_s64')
    # 'CUDA_VISIBLE_DEVICES=0 python /data/yue/keras-frcnn-master/test_frcnn.py -p /data/yue/pepple/pepple_test_data2_avg/ --network vgg --config_filename /data/yue/pepple/accell/ac_training_avg_blurred_32_32/weights_s64_class/config.pickle'
    whole_img_sensitivity(img_folder=orig_img_folder, test_folder=os.path.join(base_folder, 'pepple_test_data2_avg'), suffix='weights_s64_class')

    # TODO - 1scan (insitu) seems better than 3scan (insituM)
    # TODO - rerun plopped runs vs insitu (32*32 patches)
    # TODO - check augmented vs direct
    # TODO - evaluate on whole_images (sensitivity is correct, but precision is optimistic)
    # TODO - sensitivity is similar to interObserver, need to confirm precision and fscore vs interObserver
    # TODO - refactor code for cleanup

    # compare_nick_vs_yue(orig_img_folder, do_avg=True, ac_threshold=.3, pixel_lim=3)
    # compare_nick_vs_yue(orig_img_folder, do_avg=True, ac_threshold=.9, pixel_lim=3)
    compare_nick_vs_yue(orig_img_folder, do_avg=False, ac_threshold=.3, pixel_lim=3)
    compare_nick_vs_yue(orig_img_folder, do_avg=False, ac_threshold=.9, pixel_lim=3)