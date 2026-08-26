import os, json, cv2
import numpy as np
from matplotlib import pyplot as plt
from make_accell_data import get_img_predictions, in_ac_chamber, find_chamber_center, DOWNSAMPLE_RATIO, ACCELL_DIAMETER,\
    make_box_coords
from analyseCellPreds import recombine_predictions, get_true_coords, img_cols, img_rows
from evaluate_whole_image import get_review_txt_dict
from analyse_retinanet_accell_preds import get_img_name_from_patch_path

from sys import platform
if platform == "linux" or platform == "linux2":
    base_folder = '/data/yue/pepple'
    img_folder = '/data/yue/pepple/accell/segmentations'
    sep = '/'
elif platform == "win32":
    base_folder = 'z:/yue/pepple'
    img_folder = os.path.join(base_folder, 'accell', 'segmentations')
    sep = '\\'


# wrapper around sensitivity and precision functions
def prediction_metrics(labelled_file, pred_file, level, iou_threshold=0.5, score_threshold=0.05, outfile=None):
    labelled_cell_dict = parse_retinanet_predicted_coords(labelled_file)
    pred_dict = parse_retinanet_predicted_coords(pred_file)
    np.sum([len(x) for x in pred_dict.values()])    # num_predictions

    # since only predicting cells and not background wrap appropriately for calc_precision
    labelled_cell_dict_wrapped = {}
    labelled_cell_dict_wrapped['cell'] = labelled_cell_dict
    pred_dict_wrapped = {}
    pred_dict_wrapped['cell'] = pred_dict

    # iou_threshold = .1
    average_precisions, recall_max = calc_precision(labelled_cell_dict_wrapped, pred_dict_wrapped,
                                                    iou_threshold=iou_threshold, score_threshold=score_threshold,
                                                    level=level, outfile=outfile)
    return average_precisions, recall_max


# this is also default format for keras_frcnn training
def parse_retinanet_predicted_coords(file_path, scale=1):
    patch_cell_dict = {}
    with open(file_path, 'r') as fin:
        for l in fin.readlines():
            if ',,,,,' in l: continue   # empty line for training purposes
            l_toks = l.rstrip().split(',')
            if len(l_toks)==6:  # old system which didnt record score
                img_path, x1, y1, x2, y2, class_name = l_toks
                score = 1   # assume 100% for labelled or predicted
            elif len(l_toks)==7:
                img_path, x1, y1, x2, y2, score, class_name = l_toks
            img_base = img_path.split('/')[-1]
            if img_base not in patch_cell_dict:
                patch_cell_dict[img_base] = [(float(x1)/scale, float(y1)/scale, float(x2)/scale, float(y2)/scale, float(score))]
            else:
                patch_cell_dict[img_base].append([float(x1)/scale, float(y1)/scale, float(x2)/scale, float(y2)/scale, float(score)])
    fin.close()
    return patch_cell_dict


# precision calculations
def calc_precision(labelled_dict, pred_dict, iou_threshold=0.5, score_threshold=0.05, level='',
                   img_folder=os.path.join(base_folder, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid'),
                   write_output=False, outfile=None):
    average_precisions = {}

    img_tp_fp_dict = {}
    for label in ['cell']:
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        # evaluating over patches and since detected patches may be different!
        pred_keys = list(pred_dict[label].keys())
        labelled_keys = list(labelled_dict[label].keys())
        total_preds = 0
        total_preds_score = 0
        total_preds_max = 0

        # combine_keys = sorted(list(set(pred_keys + labelled_keys)))
        for key in pred_keys:
            detected_annotations = []
            detections = pred_dict[label][key] if key in pred_dict[label] else []
            detections = np.array(detections)
            annotations = labelled_dict[label][key] if key in labelled_dict[label] else []
            annotations = np.array(annotations)
            num_annotations += annotations.shape[0]

            # img_path = os.path.join(img_folder, key)
            # img = cv2.imread(img_path)
            # plot_preds(img, detections, annotations)
            total_preds += len(detections)
            detections = detections[detections[:,-1]>score_threshold,:]     # filter for score_threshold before nms
            total_preds_score += len(detections)
            max_suppressed_detections = non_max_suppression_fast(detections, overlapThresh=0.5)
            total_preds_max += len(max_suppressed_detections)
            # plot_preds(img, max_suppressed_detections, annotations)

            img_tp = 0
            img_fp = 0
            img_annot = len(annotations)
            for d in max_suppressed_detections:  # precision
            # for d in detections:
                d_score = d[4]

                scores = np.append(scores, d_score)
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                # overlaps_old = compute_overlap(np.expand_dims(d, axis=0), annotations)
                # assigned_annotation_old = np.argmax(overlaps_old, axis=1)
                # max_overlap = overlaps[0, assigned_annotation]
                overlaps = compute_overlap_new(d, annotations)
                assigned_annotation = np.argmax(overlaps, axis=0)
                max_overlap = overlaps[assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                    img_tp +=1
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    img_fp += 1
            img_tp_fp_dict[key] = {'tp':img_tp, 'fp':img_fp, 'annot':img_annot}

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

    #
    out_json = outfile.replace('.csv', '_l{}_s{}.json'.format(level, score_threshold))
    with open(out_json, 'w') as fout:  # save json
        json.dump(img_tp_fp_dict, fout)
    fout.close()

    print('num_annotations=', num_annotations, 'recall=', recall[-1], 'precision=', precision[-1], 'mAP=', average_precision)
    if outfile is not None:
        header = ['level', 'iou_threh', 'score_thresh', 'num_annotations', 'num_pred', 'tp', 'fp', 'recall', 'precision',
                'mAP']
        with open(outfile, 'a') as fout:
            vals = [level, iou_threshold, score_threshold, num_annotations, len(false_positives), true_positives[-1],
                    false_positives[-1], recall[-1], precision[-1], average_precision]
            fout.write('{}\n'.format(','.join([str(x) for x in vals])))
        fout.close()
    return average_precisions, recall[-1]


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh, visualise=False):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if boxes.shape[1]>4:
        # scores = boxes[:,4]
        # idxs = np.argsort(scores)
        idxs = np.argsort(area)     # largest boxes chosen for overlapping reasons
    else:
        idxs = np.argsort(y2)     # this original is INCORRECT

    # keep looping while some indexes still remain in the indexes  list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # d = boxes[i]
        # overlap2 = compute_overlap_new(d, boxes[idxs[:last]])
        # overlap3 = compute_overlap(np.expand_dims(d, axis=0), boxes[idxs[:last]])

        # delete all indexes from the index list that have
        overlap_idx = idxs[np.where(overlap > overlapThresh)[0]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        if visualise:   # visualise overlap
            # plt.figure(1)
            # plt.clf()# plt.scatter(x=[x1[i], x2[i]], y=[y1[i], y2[i]], c='red')
            # plt.scatter(x=[x1[overlap_idx], x2[overlap_idx]], y=[y1[overlap_idx], y2[overlap_idx]], c='blue')
            plt.figure(1)
            plt.clf()
            fake_img = np.zeros((img_rows, img_cols, 3))
            for j in idxs:  # all cells
                fake_img = cv2.rectangle(fake_img, (int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])), color=(0, 255, 0))
            for j in overlap_idx:   # overlapping
                fake_img = cv2.rectangle(fake_img, (int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])), color=(255, 0, 0))
            # current pred
            fake_img = cv2.rectangle(fake_img, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), color=(0, 0, 255))
            plt.imshow(fake_img)

    # return only the bounding boxes that were picked
    return boxes[pick]


def plot_preds(img, detections, annotations):
    plt.figure(1)
    plt.clf()
    plt.imshow(img[:,:,::-1])   # matplotlib is RGB
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2, score = detection
        x1, y1, x2, y2 = np.round(x1), np.round(y1), np.round(x2), np.round(y2)
        color = tuple(np.random.choice(range(256), size=3)/256)
        # plt.scatter(x=[x1, x2], y=[x2, y2], c=(0.1, 0.2, 0.5))
        plt.scatter(x=[x1, x2], y=[y1, y2], c=color)

    for idx, annotation in enumerate(annotations):
        x1, y1, x2, y2, _ = annotation
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        plt.scatter(x=[x1, x2], y=[y1, y2], c='green')
    return


# #  iou
# def compute_overlap(cell_coords, target_coords):
#     overlap_coords = make_overlapping_box(cell_coords, target_coords)
#     cell_area = _box_area_old(cell_coords)
#     target_area = _box_area_old(target_coords)
#     overlap_area = _box_area_old(overlap_coords)    # intersection area
#     union_area = cell_area + target_area - overlap_area
#     return overlap_area/union_area


def compute_overlap_new(d, boxes):  # overlap / labelled_area
    d_x1, d_y1, d_x2, d_y2, d_score = d

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
    xx1 = np.maximum(d_x1, x1)
    yy1 = np.maximum(d_y1, y1)
    xx2 = np.minimum(d_x2, x2)
    yy2 = np.minimum(d_y2, y2)

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area
    return overlap


# # this seems wrong!
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


def remove_exterior_wrapper(pred_dict, seg_folder=img_folder, visualise=False):
    # remove cells not in ac chambers
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(seg_folder)
    pred_dict_cleaned = {}
    for fname, fcells in pred_dict.items():
        img_idx = img_names.index('{}.png'.format(fname))
        img = raw_images[img_idx]
        scaled_img = converted_imgs[img_idx]
        img_pred = img_preds[img_idx]
        chamber_limits, mean_x, mean_y = find_chamber_center(img_pred, pred_threshold=0.5)
        cells_cleaned = []
        for fcell in fcells:
            if len(fcell)==4:
                x1, y1, x2, y2 = fcell
            elif len(fcell)==5:
                x1, y1, x2, y2, score = fcell
            elif len(fcell)==2:
                x, y=fcell
                x1, y1, x2, y2 = x, y, x, y
                # patch_size = 32
                # cell_size = ACCELL_DIAMETER
                # x1, y1, x2, y2 = make_box_coords(x, y, (patch_size, patch_size), cell_size)

            x = (x1+x2)/2
            y = (y1 + y2) / 2
            scaled_x = int(x / DOWNSAMPLE_RATIO)    # ac chamber on scaled image
            scaled_y = int(y / DOWNSAMPLE_RATIO)
            is_in_ac_chamber = in_ac_chamber(scaled_x, scaled_y, chamber_limits, patch_size=ACCELL_DIAMETER)
            if is_in_ac_chamber:
                cells_cleaned.append(fcell)
        pred_dict_cleaned[fname] = cells_cleaned

        if visualise:
            plt.figure(1)
            plt.clf()
            plt.imshow(img)
            temp = np.array(cells_cleaned)
            plt.scatter(x=temp[:,0], y=temp[:,1], c='red', s=2)

    np.sum([len(x) for x in pred_dict.values()])
    np.sum([len(x) for x in pred_dict_cleaned.values()])
    return pred_dict, pred_dict_cleaned


def remove_non_chamber_cells(pred_file, iou_threshold=0.5, score_threshold=0.05, level='', visualise=False):
    pred_dict = parse_retinanet_predicted_coords(pred_file)
    img_dict = recombine_predictions(pred_dict)
    #
    cleaned_json_file = os.path.join('cleaned_pred_{}.json'.format(level))
    if os.path.isfile(cleaned_json_file):
        pred_dict_cleaned = json.loads(open(cleaned_json_file).read())
    else:
        pred_dict, pred_dict_cleaned = remove_exterior_wrapper(img_dict)
        with open(cleaned_json_file, 'w') as fout:  # save json
            json.dump(pred_dict_cleaned, fout)
        fout.close()

    # get labelled jsons
    labelled_json_file = os.path.join('cleaned_labelled_{}.json'.format(level))
    if os.path.isfile(labelled_json_file):
        true_dict_cleaned = json.loads(open(labelled_json_file).read())
    else:
        labelled_cell_jsons = os.path.join(base_folder, 'accell', 'jsons_recentered_1scan')
        true_dict = get_true_coords(labelled_cell_jsons)
        _, true_dict_cleaned = remove_exterior_wrapper(true_dict)    # chamber limit labelled jsons
        with open(labelled_json_file , 'w') as fout:  # save json
            json.dump(true_dict_cleaned, fout)
        fout.close()

    # remove too bright or too low
    pred_dict_cleaned_orig = pred_dict_cleaned
    pred_dict_cleaned = filter_contrast_cells(pred_dict_cleaned, level=level, seg_folder=img_folder)    # also filter predicted cells by contrast
    np.sum([len(x) for x in pred_dict_cleaned_orig.values()])    # sanity check
    np.sum([len(x) for x in pred_dict_cleaned.values()])  # sanity check
    img_cell_dict_level = filter_contrast_cells(true_dict_cleaned, level=level, seg_folder=img_folder)

    # compare against yue_labelled
    review_file_yue = os.path.join(base_folder, 'accell', 'yue_seg_1scan.txt')
    labelled_dict_yue = get_review_txt_dict(review_file=review_file_yue, allow_commented=False)
    labelled_dict_yue_level = filter_contrast_cells(labelled_dict_yue, level=level, seg_folder=img_folder)

    if visualise:
        vis_folder = 'debug_pepple_test_data_retinanet_level{}_score{}'.format(level, score_threshold)
        if not os.path.isdir(vis_folder):
            os.makedirs(vis_folder)
        for img_name, pred_cells in pred_dict_cleaned.items():
            img_path = os.path.join(img_folder, '{}.png'.format(img_name))
            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(img_path)  # for color boxes
            plt.figure(1)
            plt.clf()
            plt.imshow(img)

            # labelled data
            yue_cells = labelled_dict_yue_level[img_name] if img_name in labelled_dict_yue_level else []
            yue_cells = np.array(yue_cells)
            nick_cells = img_cell_dict_level[img_name] if img_name in img_cell_dict_level else []
            nick_cells = np.array(nick_cells)

            pred_cells = np.array(pred_cells)
            pred_cells = pred_cells[pred_cells[:,-1]>score_threshold,:]
            pred_cells_max = non_max_suppression_fast(pred_cells, overlapThresh=0.5)     # max suppression

            # visualise and save
            new_img = draw_rectangle(img, pred_cells, color=(0, 0, 255))
            new_img = draw_rectangle(new_img, nick_cells, color=(255, 255, 0))
            new_img = draw_rectangle(new_img, yue_cells, color=(255, 0, 255))
            new_image_tiled = cv2.hconcat((img, new_img))
            cv2.imwrite(os.path.join(vis_folder, '{}.png'.format(img_name)), new_image_tiled)

            new_img2 = draw_rectangle(img, pred_cells_max, color=(0, 0, 255))
            new_img2 = draw_rectangle(new_img2, nick_cells, color=(255, 255, 0))
            new_img2 = draw_rectangle(new_img2, yue_cells, color=(255, 0, 255))
            new_image_tiled2 = cv2.hconcat((img, new_img2))
            cv2.imwrite(os.path.join(vis_folder, '{}_max.png'.format(img_name)), new_image_tiled2)

    1
    # average_precisions_nick_yue, recall_max_nick_yue = \
    #     calc_precision({'cell':img_cell_dict_level}, {'cell':labelled_dict_yue_level2}, iou_threshold=iou_threshold,
    #                    score_threshold=score_threshold, level=level, write_output=True, outfile='test_stats_nick_yue.csv')
    average_precisions_nick_dl, recall_max_nick_dl = \
        calc_precision({'cell':img_cell_dict_level}, {'cell':pred_dict_cleaned}, iou_threshold=iou_threshold,
                       score_threshold=score_threshold, level=level, write_output=True, outfile='test_stats_nick_dl.csv')
    average_precisions_yue_dl, recall_max_yue_dl = \
        calc_precision({'cell': labelled_dict_yue_level}, {'cell': pred_dict_cleaned}, iou_threshold=iou_threshold,
                       score_threshold=score_threshold, level=level, write_output=True, outfile='test_stats_yue_dl.csv')
    return


def draw_rectangle(img, boxes, color=(0, 0, 255), thickness=1, visualise=False):
    new_img = img.copy()
    for box in boxes:
        pt1 = (int(round(box[0])), int(round(box[1])))
        pt2 = (int(round(box[2])), int(round(box[3])))
        new_img = cv2.rectangle(new_img, pt1, pt2, color=color, thickness=thickness)
    if visualise:
        plt.imshow(new_img)
    return new_img


def filter_contrast_cells(img_cell_dict, level=1, seg_folder=img_folder, visualise=False):
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(seg_folder)

    img_cell_dict_level = {}
    for img_name, img_cells in img_cell_dict.items():
        img_idx = img_names.index('{}.png'.format(img_name))
        img = raw_images[img_idx]
        scaled_img = converted_imgs[img_idx]
        img_pred = img_preds[img_idx]
        img_mean = np.mean(img)

        filtered_cells = []
        for cell in img_cells:
            if len(cell)==4:
                x1, y1, x2, y2 = cell
                score = 1   # this is labelled
            elif len(cell)==5:
                x1, y1, x2, y2, score = cell
            elif len(cell)==2:
                x, y = cell
                cell_size = ACCELL_DIAMETER
                x1, y1, x2, y2 = make_box_coords((x, y), img.shape, cell_size)
                score = 1   # this is labelled

            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
            if visualise:
                plt.figure(1)
                plt.clf()
                plt.imshow(img)
                plt.scatter(x=[x1, x2], y=[y1, y2], c='red')

            cell_patch = img[y1:y2 + 1, x1:x2 + 1]  # this is correct
            # cell_patch = img[y1:y2, x1:x2]    # undercounting compared to area formula
            cell_mean = np.mean(cell_patch)

            if cell_mean/img_mean>level:
                # filtered_cells.append(cell)
                filtered_cells.append((x1, y1, x2, y2, score))     # fake bounding box coords if labelled cells
            else:   # ignore - not bright enough
                1
        img_cell_dict_level[img_name] = filtered_cells
    return img_cell_dict_level


def plot_surface(result_file, metric='mAP'):
    data = np.genfromtxt(result_file, delimiter=',', skip_header=1)
    header = ['level', 'iou_threh', 'score_thresh', 'num_annotations', 'num_pred', 'tp', 'fp', 'recall', 'precision',
            'mAP']

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(1)
    plt.clf()
    # ax = fig.gca(projection='3d')
    ax = Axes3D(fig)

    # num_levels = 7
    # X = np.linspace(1, 2.5, num_levels)
    # num_levels = 5
    # X = np.linspace(1.5, 2.5, num_levels)
    num_levels = 6
    X = np.linspace(1, 2.25, num_levels)
    num_contrasts = 12
    # Y = np.linspace(0.05, 0.6, num_contrasts)
    Y = np.array([.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6])
    X2, Y2 = np.meshgrid(X, Y)
    Z = np.zeros(X2.shape)

    recall_index = header.index('recall')
    prec_index = header.index('precision')
    if metric in header:
        metric_index = header.index(metric)
    else:
        metric_index = -1
    for idx, level in enumerate(X):
        for jdx, contrast in enumerate(Y):
            cur_data = data[np.logical_and(data[:,0]==level, data[:,2]==contrast),:]
            # print(level, contrast, metric_index, cur_data)
            if metric_index>-1:
                Z[jdx, idx] = cur_data[:, metric_index]
            else:   # fscore
                cur_recall = cur_data[:, recall_index]
                cur_prec = cur_data[:, prec_index]
                fscore = 2 * (cur_recall*cur_prec)/(cur_recall+cur_prec+np.finfo(np.float64).eps)
                Z[jdx, idx] = fscore

    from matplotlib import cm
    surf = ax.plot_surface(X2, Y2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('contrastLevels')
    plt.ylabel('scoreThresholds')
    ax.set_zlabel(metric)
    plt.title('Surface for {}'.format(metric))
    save_file = result_file.replace('.csv', '_{}.png'.format(metric))
    plt.savefig(save_file, bbox_inches='tight')
    return


def understand_whole_img_patches(patch_folder=os.path.join(base_folder, 'pepple_test_data'), img_folder=img_folder):
    patch_files = [x for x in sorted(os.listdir(patch_folder)) if '.png' in x]
    # dictionary of patches by img
    img_patch_dict = {}
    for patch_file in patch_files:
        img_name, x, y = get_img_name_from_patch_path(patch_file)
        if img_name in img_patch_dict:
            img_patch_dict[img_name].append((x,y))
        else:
            img_patch_dict[img_name] = [(x,y)]

    # now visualise the boxes
    for img_name in img_patch_dict.keys():
        img_data = np.array(img_patch_dict[img_name])

        img_path = os.path.join(img_folder, '{}.png'.format(img_name))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.scatter(x=img_data[:,0], y=img_data[:,1], c='red', s=5)
    return


if __name__ == '__main__':
    # levels = [1.5, 1.75, 2.0, 2.25, 2.5]
    # level = levels[0]
    # # epoch = 39
    # epoch = 75
    # prediction_metrics(labelled_file=os.path.join(base_folder, 'accell', 'ac_training_insitu_nickHypo_new', 'valid', 'valid_coords_{}.txt'.format(level)),
    #                    pred_file=os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}'.format(level), 'pred_weights', 'coords_pred_{}.txt'.format(epoch)))

    # # better experiment
    levels = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    levels = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    levels = [1.0, 1.25, 1.5]
    # levels = [1.5, 1.75, 2.0, 2.25, 2.5]    # for /data/yue/pepple/accell/ac_training_insitu_nickHypo_new/valid/
    score_thresholds = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6]

    for level in levels:
        epoch = 20
        # epoch = 30
        # if level==2.5:
        #     epoch = 23
        # level = levels[1]
        for score_threshold in score_thresholds:
            # prediction_metrics(
            #     labelled_file=os.path.join(base_folder, 'accell', 'ac_training_insitu_nickHypo_new2', 'valid', 'valid_coords_{}.txt'.format(level)),
            #     pred_file=os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}_new'.format(level), 'pred_weights', 'coords_pred_{}.txt'.format(epoch)),
            #     level=level, iou_threshold=.5, score_threshold=score_threshold)
            # prediction_metrics(
            #     labelled_file=os.path.join(base_folder, 'accell', 'ac_training_insitu_nickHypo_new', 'valid', 'valid_coords_{}.txt'.format(level)),
            #     pred_file=os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}'.format(level), 'pred_weights', 'coords_pred_{}.txt'.format(epoch)),
            #     level=level, iou_threshold=.5, score_threshold=score_threshold, outfile='valid_log_5by5_nms.csv')
            # prediction_metrics(
            #     labelled_file=os.path.join(base_folder, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_new_{}.txt'.format(level)),
            #     pred_file=os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}_new3'.format(level), 'coords_pred_{}.txt'.format(epoch)),
            #     level=level, iou_threshold=.5, score_threshold=score_threshold, outfile='valid_log_new3.csv')   # actually dynamic not 5by5
            prediction_metrics(
                labelled_file=os.path.join(base_folder, 'accell', 'ac_training_insitu_nickHypo_new3', 'valid', 'valid_coords_fixed_{}.txt'.format(level)),
                pred_file=os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}_new3_5by5'.format(level), 'coords_pred_{}.txt'.format(epoch)),
                level=level, iou_threshold=.5, score_threshold=score_threshold, outfile='valid_log_new3_5by5.csv')   # actually dynamic not 5by5

    # # understand surface of results
    # # result_file = os.path.join(base_folder, 'valid_log_5by5_nms.csv')
    # result_file = os.path.join('valid_log_5by5_nms_new.csv')    # actually dynamic not 5by5
    # plot_surface(result_file=result_file, metric='mAP')
    # plot_surface(result_file=result_file, metric='recall')
    # plot_surface(result_file=result_file, metric='precision')
    # plot_surface(result_file=result_file, metric='fscore')

    iou_thresh = 0.5
    iou_thresh = 0.01   # minimum overlap - sometimes predictions are tiny
    # levels = [1.5, 1.75, 2.0, 2.25, 2.5]
    for level in levels:
        epoch = 20
        # epoch = 30
        # if level==2.5:
        #     epoch = 23
        for score_threshold in score_thresholds:
            pred_file = os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}_new'.format(level), 'pred_weights', 'coords_pred_whole_{}.txt'.format(epoch))
            pred_file = os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}'.format(level), 'pred_weights', 'coords_pred_whole_{}.txt'.format(epoch))
            pred_file = os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}_new3'.format(level), 'coords_pred_whole_{}.txt'.format(epoch))
            pred_file = os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}_new3_5by5'.format(level), 'coords_pred_whole_{}.txt'.format(epoch))
            remove_non_chamber_cells(pred_file=pred_file, iou_threshold=iou_thresh, score_threshold=score_threshold, level=level, visualise=False)
    import sys
    sys.exit()
    # # best and worst images per contrast level and score_threshold
    # level = 2.0
    # score_threshold = 0.25
    # img_pred_json = os.path.join('post_nms_fix_post_contrast_level_fix', 'test_stats_nick_dll{}_s{}.json'.format(level, score_threshold))
    # img_tp_fp_dict = json.loads(open(img_pred_json).read())
    # for key, vals in img_tp_fp_dict.items():
    #     print(key, vals['tp'], vals['fp'], vals['annot'])

    # result_file = os.path.join(base_folder, 'test_stats_nick_dl_max.csv')
    result_file = os.path.join('post_nms_fix_post_contrast_level_fix', 'test_stats_nick_dl.csv')
    result_file = os.path.join('test_stats_nick_dl.csv')
    # result_file = os.path.join('test_stats_nick_dl_max_5by5.csv')
    # result_file = os.path.join('post_nms_fix_post_contrast_level_fix', 'test_stats_yue_dl.csv')
    plot_surface(result_file=result_file, metric='mAP')
    plot_surface(result_file=result_file, metric='recall')
    plot_surface(result_file=result_file, metric='precision')
    plot_surface(result_file=result_file, metric='fscore')

    # # understand_whole_img_patches()
    # level = 1.0
    # epoch = 30
    # score_threshold = .3
    # pred_file = os.path.join(base_folder, 'accell', 'retinanet_keras', 'thresh{}_new'.format(level), 'pred_weights',
    #                          'coords_pred_whole_{}.txt'.format(epoch))
    # remove_non_chamber_cells(pred_file=pred_file, iou_threshold=0.5, score_threshold=score_threshold, level=level, visualise=True)