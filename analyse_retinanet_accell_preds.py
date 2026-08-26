import os, json, cv2
import numpy as np
from matplotlib import pyplot as plt

from sys import platform
if platform == "linux" or platform == "linux2":
    pepple_folder = '/data/yue/pepple'
    sep = '/'
elif platform == "win32":
    pepple_folder = 'z:/yue/pepple'
    sep = '\\'
patch_path_sep = '/'
base_folder = os.path.join(pepple_folder, 'accell')
orig_img_folder = os.path.join(base_folder, 'segmentations')
empty_img_folder = os.path.join(base_folder, 'empty_segmentations')
json_folder = os.path.join(base_folder, 'jsons')

from make_accell_data import get_img_predictions, DOWNSAMPLE_RATIO, range_overlap, make_box_coords, RAW_IMG_ROWS, RAW_IMG_COLS
from predict_image_cells import any_middle_stripes, calc_img_chamber_size
from evaluate_whole_image import get_true_coords_by_class


def parse_retinanet_preds(pred_path, visualise=False):
    predicted_dict = {}
    with open(pred_path) as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(",")
            fname, x1, y1, x2, y2, obj_class = l_toks
            obj_loc_class = (float(x1.strip()), float(y1), float(x2), float(y2), int(obj_class))
            if fname not in predicted_dict:
                predicted_dict[fname] = [obj_loc_class]
            else:
                predicted_dict[fname].append(obj_loc_class)
            if visualise:
                plot_img_coords(fname, predicted_dict[fname])
    return predicted_dict


def combine_preds(pred_dict):
    img_preds_dict = {}
    for patch_path, patch_preds in pred_dict.items():
        img_name, x, y = get_img_name_from_patch_path(patch_path)
        for patch_pred in patch_preds:  # adjust preds
            x1, y1, x2, y2, obj_class = patch_pred
            adj_pred = (x1+x, y1+y, x2+x, y2+y, obj_class)
            if img_name not in img_preds_dict:
                img_preds_dict[img_name] = [adj_pred]
            else:
                img_preds_dict[img_name].append(adj_pred)
    return img_preds_dict


def get_img_name_from_patch_path(patch_path):
    patch_toks = patch_path.split(patch_path_sep)
    # img_folder = sep.join(patch_toks[:-1])
    patch_name = patch_toks[-1]
    patch_name_toks = patch_name.replace('.png', '').split('_')
    y = patch_name_toks[-2].replace('h', '')
    x = patch_name_toks[-1].replace('w', '')
    img_name = '_'.join(patch_name_toks[:-2])
    return img_name, int(x), int(y)


def visualise_vs_truth(preds_file, json_folder):
    true_dict = get_true_coords(json_folder)    # read coords file into dictionary
    # adjust to box coords for precision calculations
    true_dict_box = {}
    for img_name, cell_locs in true_dict.items():
        true_dict_box[img_name] = []
        for cell_loc in cell_locs:
            cell_box = make_box_coords(cell_loc, img_shape=(RAW_IMG_ROWS, RAW_IMG_COLS))
            true_dict_box[img_name].append(cell_box)

    preds_folder = sep.join(preds_file.split(sep)[:-1])
    preds_epoch = preds_file.split('_')[-1].replace('.txt', '')
    preds_folder = os.path.join(preds_folder, 'preds_{}'.format(preds_epoch))
    if not os.path.isdir(preds_folder):
        os.makedirs(preds_folder)

    preds_dict = parse_retinanet_preds(preds_file)  # parse preds
    combined_dict = combine_preds(preds_dict)      # combine_preds

    # # visualise preds on images
    # for img_name, img_obj_preds in combined_dict.items():  # loop over preds as more likely over predictions
    #     img_path = os.path.join(orig_img_folder, '{}.png'.format(img_name))
    #     # plot_img_coords(img_path, img_obj_preds, true_dict[img_name])
    #     img, img_cp = visualise_img_coords(img_path, img_obj_preds, true_dict[img_name])
    #     img_stack = cv2.hconcat((img, img_cp))
    #     cv2.imwrite(os.path.join(preds_folder, '{}.png'.format(img_name)), img_stack)

    # take out ac chamber
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(orig_img_folder)
    preds_dict = remove_exterior_cells(combined_dict, raw_images, img_names, img_preds, preds_file=preds_file,
                                       use_contour_fill=True, mid_line=True, debug_images=False, visualise=False)
    # wrap up
    labelled_cell_dict_wrapped = {}
    labelled_cell_dict_wrapped['cell'] = true_dict_box
    pred_dict_wrapped = {}
    pred_dict_wrapped['cell'] = preds_dict
    average_precisions, max_recall = calc_precision(labelled_cell_dict_wrapped, pred_dict_wrapped, iou_threshold=0.2)

    # class_int2name_dict, class_name2int_dict = get_class_map()
    # pred_dict_wrapped = {}
    # for img_name, cells in preds_dict.items():
    #     for cell in cells:
    #         cell_class = class_name2int_dict[cell[-1]]
    #         if cell_class in pred_dict_wrapped:
    #             if img_name in pred_dict_wrapped[cell_class]:
    #                 pred_dict_wrapped[cell_class][img_name].append(cell)
    #             else:
    #                 pred_dict_wrapped[cell_class][img_name] = [cell]
    #         else:
    #             pred_dict_wrapped[cell_class] = {}  # init
    #             pred_dict_wrapped[cell_class][img_name] = [cell]
    #
    # # json_folder doesnt have necessary classification of cells
    # true_dict_by_class = get_true_coords_by_class(true_dict_box, raw_images, img_names, visualise=False)
    # average_precisions, max_recall = calc_precision(labelled_cell_dict_wrapped, pred_dict_wrapped, iou_threshold=0.5)
    return true_dict, preds_dict, combined_dict, average_precisions, max_recall


def get_class_map(map_file=os.path.join(base_folder, 'cell_classes.txt')):
    class_int2name_dict = {}
    class_name2int_dict = {}
    with open(map_file, 'r') as fin:
        for l in fin.readlines():
            l_toks = l.rstrip().split(',')
            class_name, class_int = l_toks
            class_int2name_dict[int(class_int)] = class_name
            class_name2int_dict[class_name] = int(class_int)
    fin.close()
    return class_int2name_dict, class_name2int_dict


# np.int64 etc not JSON serializable
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def remove_exterior_cells(pred_dict, imgs, img_names, img_preds, preds_file,
                          use_contour_fill=True, mid_line=False, debug_images=True, visualise=False):
    pred_toks = preds_file.split(sep)
    out_folder = sep.join(pred_toks[:-1])
    epoch_num = pred_toks[-1].split('_')[-1].replace('.txt', '')
    output_type = 'adj_cells_c{}_m{}_e{}'.format(int(use_contour_fill), int(mid_line), epoch_num)
    out_file = os.path.join(out_folder, '{}.json'.format(output_type))
    if os.path.isfile(out_file):
        fin = open(out_file).read()
        pred_dict_ac_adjusted = json.loads(fin)
        return pred_dict_ac_adjusted

    pred_dict_ac_adjusted = {}
    for key, cells in pred_dict.items():    # already combined
        key_base = key.split(patch_path_sep)[-1]
        cur_index = img_names.index('{}.png'.format(key_base))
        img = imgs[cur_index,]
        mask = img_preds[cur_index,]
        chamber_limits, chamber_size, mean_x, mean_y = calc_img_chamber_size(mask, use_contour_fill=use_contour_fill, pred_threshold=.8, img=img)
        if mid_line:    # can also use human_accell/analyse_results.py/any_middle_stripes (detects top of chamber then midline)
            mid_limits, mid_min, mid_max = any_middle_stripes(img, avg_period=10, intensity_threshold=180)
        else:
            mid_min, mid_max = None, None

        # already combined cell coords
        for idx, cell in enumerate(cells):
            x1, y1, x2, y2, obj_class = cell    # uses older keras_retinanet.eval with No score
            in_ac_flag = is_cell_in_ac(cell, chamber_limits, mean_x, mean_y, conservative=0, no_mid=int(mid_line), mid_min=mid_min, mid_max=mid_max, img=img)
            if visualise:
                plt.figure(1), plt.clf()
                plt.imshow(img), plt.scatter(x=[x1, x2], y=[y1, y2], c='red')
            if in_ac_flag:
                if debug_images:
                    img_cp = overlay_cells_on_img(img, [cell], color=(255, 0, 0))
                    debug_folder = os.path.join(out_folder, output_type)
                    if not os.path.isdir(debug_folder): os.makedirs(debug_folder)
                    img_out_path = os.path.join(debug_folder, '{}_{}.png'.format(key_base, idx))
                    cv2.imwrite(img_out_path, img_cp)

                if key_base in pred_dict_ac_adjusted:
                    pred_dict_ac_adjusted[key_base].append(cell)
                else:
                    pred_dict_ac_adjusted[key_base] = [cell]
            else:
                print('not in ac', key_base, cell)
                if debug_images:
                    img_cp = overlay_cells_on_img(img, [cell], color=(255, 0, 0))
                    debug_folder = os.path.join(out_folder, '{}_missed'.format(output_type))
                    if not os.path.isdir(debug_folder): os.makedirs(debug_folder)
                    img_out_path = os.path.join(debug_folder, '{}_{}.png'.format(key_base, idx))
                    cv2.imwrite(img_out_path, img_cp)
    with open(out_file, 'w') as fout:
        json.dump(pred_dict_ac_adjusted, fout, cls=MyEncoder)
    fout.close()
    print('num in chamber', np.sum([len(val) for val in pred_dict_ac_adjusted.values()]))
    return pred_dict_ac_adjusted


def calc_img_chamber_size(mask, img=None, use_contour_fill=True, pred_threshold=.9, visualise=False):
    if use_contour_fill:
        chamber_limits = get_chamber_contour(img, mask)
    else:
        # pred_threshold = 0.9    # very conservative threshold
        chamber_limits = np.argwhere(mask > pred_threshold)    # n*2 where either 1st/2nd coord below pred_threshold
    mean_x, mean_y = get_chamber_center(chamber_limits)
    chamber_size = len(chamber_limits)

    if visualise:
        plt.figure(1)
        plt.imshow(img[::2, ::2])   # every other pixel downsampling for visual purposes
        plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='yellow', s=1)
        plt.scatter(x=mean_x, y=mean_y, c='green', s=1)
        img_cp = overlay_mask(img[::2,::2].astype(np.float64), mask, alpha=.4, beta=1)
        plt.figure(2), plt.clf(), plt.imshow(img_cp)

    return chamber_limits, int(chamber_size), mean_x, mean_y


def get_chamber_contour(img, mask, use_max=True, visualise=False):
    im2, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_ds = img[::2,::2]
    # img_ds = downsample_img(img, scale=1 / DOWNSAMPLE_RATIO)
    # img_ds_3channel = np.repeat(np.expand_dims(img_ds, axis=2), 3, axis=2)
    if use_max:  # get max contour and close
        max_contour_area = 0
        max_contour = None
        for c in contours:
            cur_area = cv2.contourArea(c)
            if cur_area>max_contour_area:
                max_contour = c
                max_contour_area = cur_area
        temp = cv2.fillPoly(img_ds/255, pts=[max_contour], color=(255))  # not destructive of img_ds
        filled_img = np.argwhere(temp>1)
        # temp = cv2.fillPoly(img_ds_3channel, pts=[max_contour], color=(0, 0, 255))  # not destructive of img_ds
        # filled_img = np.argwhere(np.logical_and(temp[:,:,0]==0, np.logical_and(temp[:,:,1]==0, temp[:,:,2]==255)))
        if visualise:
            plt.figure(2), plt.clf(), plt.imshow(temp)
            plt.figure(3), plt.clf(), plt.scatter(x=filled_img[:,1], y=filled_img[:,0])
    else:   # all contours - this is more holely!
        # temp = cv2.fillPoly(img_ds_3channel, pts=contours, color=(0, 0, 255))  # not destructive of img_ds
        temp = cv2.fillPoly(img_ds / 255, pts=contours, color=(255))  # not destructive of img_ds
        filled_img = np.argwhere(temp > 1)
        if visualise:
            plt.figure(4), plt.clf(), plt.imshow(temp)
            plt.figure(5), plt.clf(), plt.scatter(x=filled_img[:, 1], y=filled_img[:, 0])

    if visualise:  # 3 channel to see color properly - just use 1st channel!
        temp = cv2.drawContours(img_ds, contours, -1, (255, 0, 0), 3)
        plt.figure(1), plt.clf(), plt.imshow(temp)
    return filled_img


def is_cell_in_ac(tiff_cell, chamber_limits, mean_x, mean_y, conservative=0, no_mid=0, mid_min=None, mid_max=None, img=None):
    if len(tiff_cell)==4:
        x1, y1, x2, y2 = tiff_cell  # coords on 1024*1000
    elif len(tiff_cell)==5:
        x1, y1, x2, y2, prob = tiff_cell  # coords on 1024*1000

    no_mid_stripe = True
    if no_mid and mid_min is not None and mid_max is not None:  # not in middle stripe -> True
        # no_mid_stripe = not range_overlap(x1, x2, mid_min, mid_max)
        no_mid_stripe = not range_overlap(x1, x2, mid_min-conservative, mid_max+conservative)
        # no_mid_stripe = True if (x1 < mid_min or x1 > mid_max) and (x2 < mid_min or x2 > mid_max) else False

    top_in_ac = in_chamber((x1,y1), chamber_limits, mean_x, mean_y, conservative=conservative, img=img)
    bottom_in_ac = in_chamber((x2,y2), chamber_limits, mean_x, mean_y, conservative=conservative, img=img)
    return no_mid_stripe and top_in_ac and bottom_in_ac


def in_chamber(coord, chamber_coords, mean_x, mean_y, conservative=0, img=None, visualise=False):
    x, y = coord    # 1024*1000 scale
    x1 = round(x/DOWNSAMPLE_RATIO)
    y1 = round(y/DOWNSAMPLE_RATIO)

    if conservative:    # move x1, y1 'outside' the chamber to test if right on edge
        centralize_factor = conservative    # override meaning of conservative
        try:
            # print(mean_x, x1, mean_y, y1)
            x1 += centralize_factor * np.sign(mean_x - x1) * -1     # -1 to make x,y not in chamber_coords
            y1 += centralize_factor * np.sign(mean_y - y1) * -1     # -1 for opposite direction
        except:
            print(mean_x, x1, mean_y, y1)

    if img is not None and visualise:  # visualise
        plt.figure(1)
        plt.clf()
        plt.imshow(img[::2, ::2])   # every other pixel downsampling for visual purposes
        plt.scatter(x=chamber_coords[:, 1], y=chamber_coords[:, 0], c='yellow', s=1)    # NB. y in col 0
        plt.scatter(x=mean_x, y=mean_y, c='green', s=2)
        plt.scatter(x=round(coord[0]/DOWNSAMPLE_RATIO), y=round(coord[1]/DOWNSAMPLE_RATIO), c='blue', s=2)
        plt.scatter(x=x1, y=y1, c='red', s=2)

    if [y1, x1] in chamber_coords.tolist():     # NB coords swapped and nearest point in chamber_coords (512*500)
        return True
    else:
        return False


def get_chamber_center(chamber_coords):
    if len(chamber_coords) > 0:
        # better way to find center
        mean_x = np.mean(chamber_coords[:, 1])  # careful about meaning of coordinates: 1st coord is y-axis
        # mean_y = np.percentile(chamber_limits[:, 0], q=25)
        mean_y = np.mean(chamber_coords[chamber_coords[:, 1] == int(mean_x), 0])  # mid-y for central x
        if np.isnan(mean_y):    # in case middle is fuzzy
            mean_y1 = np.mean(chamber_coords[chamber_coords[:, 1] == int(mean_x-20), 0])  # mid-y for central x
            mean_y2 = np.mean(chamber_coords[chamber_coords[:, 1] == int(mean_x+20), 0])  # mid-y for central x
            mean_y = (mean_y1+mean_y2)/2
    else:
        mean_x = 0
        mean_y = 0
    return mean_x, mean_y


def overlay_mask(img, mask, alpha=.4, beta=1):
    overlaid_img = cv2.addWeighted(img, alpha, mask, beta, 0)
    return overlaid_img


# copied from analyseCellPreds
def get_coords(coord_file):
    fin = open(coord_file).read()
    json_data = json.loads(fin)
    return json_data


def overlay_cells_on_img(img, cells, color=(0, 0, 255)):
    img_cp = img.copy()
    for cell in cells:
        if len(cell)==4:
            x1, y1, x2, y2 = cell
            x_mean, y_mean = int((x1+x2)/2), int((y1+y2)/2)
        elif len(cell)==2:
            x_mean, y_mean = cell
        elif len(cell)==5:
            x1, y1, x2, y2, _ = cell
            x_mean, y_mean = int((x1+x2)/2), int((y1+y2)/2)
        cv2.circle(img_cp, (x_mean, y_mean), 2, color, 2)
    return img_cp


# for outputting images use cv2
def visualise_img_coords(img_path, file_coords, true_coords=None):
    if platform == "win32":
        img_path = img_path.replace('/data/yue', 'z:/yue')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_3channel = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
    img_cp = overlay_cells_on_img(img_3channel, file_coords, color=(0, 0, 255))  # red
    if true_coords is not None:
        img_cp = overlay_cells_on_img(img_cp, true_coords, color=(0, 255, 0))   # green
    return img_3channel, img_cp


# matplotlib version
def plot_img_coords(img_path, file_coords, true_coords=None, s=3):
    if platform == "win32":
        img_path = img_path.replace('/data/yue', 'z:/yue')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(1), plt.clf()
    plt.imshow(img)
    file_coords_np = np.asarray(file_coords)
    plt.scatter(x=file_coords_np[:, 0], y=file_coords_np[:, 1], c='red', s=s)
    if file_coords_np.shape[1]>=4:
        plt.scatter(x=file_coords_np[:, 2], y=file_coords_np[:, 3], c='red', s=s)

    if true_coords is not None:
        true_coords_np = np.asarray(true_coords)
        plt.scatter(x=true_coords_np[:, 0], y=true_coords_np[:, 1], c='lime', s=s)
        if true_coords_np.shape[1]>=4:
            plt.scatter(x=true_coords_np[:, 2], y=true_coords_np[:, 3], c='lime', s=s)
    return img


def get_true_coords(folder, visualise=False):
    json_files = [x for x in os.listdir(folder) if '.json' in x]
    coord_dict = {}
    for json_file in json_files:
        fname = json_file.replace('.json', '')
        file_data = get_coords(os.path.join(folder, json_file))
        file_coords = []
        for coords in file_data:
            x_coords = coords['mousex']
            y_coords = coords['mousey']
            for idx, x_coord in enumerate(x_coords):
                file_coords.append((x_coord, y_coords[idx]))
        coord_dict[fname] = file_coords

        if visualise:
            plot_img_coords(orig_img_folder, fname, file_coords)
    return coord_dict


# metric functions
def _box_area_old(box_coords):
    if len(box_coords)==5:
        x1, y1, x2, y2, _ = box_coords
    else:
        x1, y1, x2, y2 = box_coords
    return float((x2-x1+1)*(y2-y1+1))


def make_overlapping_box(box_coord_1, box_coord_anchor):
    if len(box_coord_1)==5:
        x1_low, y1_low, x1_upper, y1_upper, _ = box_coord_1
    else:
        x1_low, y1_low, x1_upper, y1_upper = box_coord_1
    x2_low, y2_low, x2_upper, y2_upper = box_coord_anchor

    overlap_x1 = max([x1_low, x2_low])
    overlap_x2 = min([x1_upper, x2_upper])
    overlap_y1 = max([y1_low, y2_low])
    overlap_y2 = min([y1_upper, y2_upper])
    return (overlap_x1, overlap_y1, overlap_x2, overlap_y2)


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


# #  iou
# def compute_overlap(cell_coords, target_coords):
#     overlap_coords = make_overlapping_box(cell_coords, target_coords)
#     cell_area = _box_area_old(cell_coords)
#     target_area = _box_area_old(target_coords)
#     overlap_area = _box_area_old(overlap_coords)    # intersection area
#     union_area = cell_area + target_area - overlap_area
#     return overlap_area/union_area


# precision calculations
def calc_precision(labelled_dict, pred_dict, iou_threshold=0.5):
    average_precisions = {}

    for label in labelled_dict.keys():  # class
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        # evaluating over patches and since detected patches may be different!
        combine_keys = list(set(list(pred_dict[label].keys()) + list(labelled_dict[label].keys())))
        for key in combine_keys:
            detected_annotations = []
            detections = pred_dict[label][key] if key in pred_dict[label] else []
            detections = np.array(detections)
            annotations = labelled_dict[label][key] if key in labelled_dict[label] else []
            annotations = np.array(annotations)
            num_annotations += annotations.shape[0]
            for d in detections:  # precision
                if len(d)>=5:
                    scores = np.append(scores, d[4])
                else:
                    scores = np.append(scores, 1.0)

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations   # sensitivity
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
    return average_precisions, recall[-1]


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


if __name__ == '__main__':
    preds_files = [os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_keras_resnet50_preds', 'avg_1class', 'resnet50_25.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_keras_resnet50_preds', 'avg_1class', 'resnet50_43.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_keras_resnet50_preds', 'avg_1class_with_fp', 'resnet50_25.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_keras_resnet50_preds', 'avg_1class_with_fp', 'resnet50_43.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_keras_resnet50_preds', 'avg_3class', 'resnet50_25.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_keras_resnet50_preds', 'avg_3class', 'resnet50_43.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_resnet50_preds', 'resnet50_25.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_resnet50_preds', 'resnet50_50.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_resnet50_preds', 'resnet50_75.txt'),
                   os.path.join(pepple_folder, 'pepple_test_data2_avg', 'retinanet_resnet50_preds', 'resnet50_85.txt')]
    for preds_file in preds_files:
        # parse_retinanet_preds(preds_file)  # parse preds file
        ## visualise vs truth and save output
        json_folder = os.path.join(base_folder, 'jsons_recentered')
        # json_folder = os.path.join(base_folder, 'jsons_recentered_1scan')
        visualise_vs_truth(preds_file, json_folder=json_folder)
    ## try different classes
    ## compute stats using IOU
    ## remove chamber using acseg and compute stats using IOU
