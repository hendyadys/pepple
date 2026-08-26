import numpy as np
import os, json
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

from make_accell_data import get_img_predictions, parse_img_base_name, get_patch_stats, recenter_cell, \
    DOWNSAMPLE_RATIO, ACCELL_DIAMETER, RAW_IMG_COLS
from evaluate_whole_image import check_neighbouring_cells_from_dict, log_sensitivity_precision, avg_images, \
    get_true_coords, check_neighbouring_cells, get_valid_images, get_true_dict_for_validation, get_true_coords_by_class, \
    constrain_by_ac_chamber, get_review_txt_dict, save_imgs


def get_imgs(folder):
    img_names = [x for x in sorted(os.listdir(folder)) if '.png' in x]
    imgs = []
    for img_name in img_names:
        cur_img = cv2.imread(os.path.join(folder, img_name))    # need color for labelled cells and hyperion
        imgs.append(cur_img)
    return img_names, imgs


def get_labelled_cells(img, color=(36, 28, 237), visualise=False):
    # cell_locs = np.nonzero(img[:,:,0]!=img[:,:,1])
    # cell_locs_argwhere = np.argwhere(img[:, :, 0] != img[:, :, 1])
    cell_locs_argwhere = np.argwhere(np.logical_and(np.logical_and(img[:,:,0]==color[0], img[:,:,1]==color[1]), img[:,:,2]==color[2]))

    if visualise:
        plt.figure(1)
        plt.clf()
        plt.imshow(img)     # image has old color already
        plt.scatter(x=cell_locs_argwhere[:,1], y=cell_locs_argwhere[:,0], s=2, c='red')
    return cell_locs_argwhere


def get_all_labelled_cells(img_names, imgs, do_avg=False, visualise=False):
    json_file = os.path.join(kathryn_folder, 'non_dupe_cells.json')
    if os.path.isfile(json_file):
        fin = open(json_file).read()
        labelled_cells_dict = json.loads(fin)
        return labelled_cells_dict

    # get red points
    labelled_cells = {}
    for idx, img_name in enumerate(img_names):
        # if idx>0: continue  # debug HACK
        img = imgs[idx]
        cell_locs = get_labelled_cells(img)
        # meaning of x,y different in np.argwhere see plt.scatter(x=chamber_limits[:,1], y=chamber_limits[:,0]) as example
        temp = cell_locs.copy()
        temp[:,0] = cell_locs[:,1]
        temp[:,1] = cell_locs[:,0]
        labelled_cells[img_name] = temp

    # remove duplicates
    cell_diam = ACCELL_DIAMETER+2  # some of kathryn's markers are big
    labelled_cells_dict, duplicate_cells_dict = check_neighbouring_cells_from_dict(labelled_cells, cell_diam=cell_diam)
    labelled_cells_dict_serializable = {}
    for key, val in labelled_cells_dict.items():
        labelled_cells_dict_serializable[key] = np.array(val).tolist()
    with open(json_file, 'w') as fout:
        json.dump(labelled_cells_dict_serializable, fout)
    fout.close()

    if visualise:
        for idx, img_name in enumerate(img_names):
            # idx=20
            # img_name = img_names[idx]
            plt.figure(1)
            plt.clf()
            plt.imshow(imgs[idx])   # already has labels
            cur_coords = np.asarray(labelled_cells_dict[img_name])
            plt.scatter(x=cur_coords[:,0], y=cur_coords[:,1], s=1, color='red')
    return labelled_cells_dict


def recenter_cells_from_dict(raw_images, img_names, img_preds, img_cells_dict, seg_folder, do_avg=True):
    if do_avg:
        recentered_folder = os.path.join(seg_folder, 'jsons_recentered')
    else:
        recentered_folder = os.path.join(seg_folder, 'jsons_recentered_1scan')
    if not os.path.isdir(recentered_folder):
        os.makedirs(recentered_folder)

    cell_size = ACCELL_DIAMETER
    all_cell_data = np.ndarray((0, cell_size, cell_size), dtype=np.uint8)
    cell_labels = []
    for idx, img_name in enumerate(img_names):  # get accells for segmented image
        cell_labels.append(img_name)

        recentered_coords = []     # for output
        sample_name = parse_img_base_name(img_name)
        coords = img_cells_dict[img_name]
        # json_path = '{}/{}.json'.format(seg_folder, sample_name)
        # coords = get_coords(json_path)
        cur_image = raw_images[idx]  # since accell segmentation on 1024*1000 png
        cur_pred = img_preds[idx]     # NB - on smaller scale 512*512
        img_intensity_mean, img_intensity_std, img_shape = get_patch_stats(cur_image)  # img_stats - probably less relevant than chamber_stats

        img_cell_data = np.zeros((len(coords), cell_size, cell_size), dtype=np.uint8)
        for jdx, coord in enumerate(coords):
            cur_x, cur_y = coord
            cur_cell, cell_coords, old_cell = recenter_cell(cur_image, cur_x, cur_y, img_shape, img_pred=cur_pred, img_name=sample_name)
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


def compare_all_users(img_folder, do_avg=True, ac_threshold=.3, pixel_lim=3):
    if do_avg:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered')
        labelled_kathryn_jsons = os.path.join(kathryn_folder, 'jsons_recentered')
    else:
        labelled_cell_jsons = os.path.join('accell', 'jsons_recentered_1scan')
        labelled_kathryn_jsons = os.path.join(kathryn_folder, 'jsons_recentered_1scan')

    # for visualisation and checking preds actually in chamber
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder=img_folder)
    # dont need avg to visualise or get chamber coords! so skip do_avg for speed!
    if do_avg:
        raw_images_old = np.copy(raw_images)
        raw_images, _ = avg_images(img_names)

    # get nick labelled data
    true_dict = get_true_coords(labelled_cell_jsons)    # this grabs labels from nick, kathryn and leslie on single scans
    print('all labelled', np.sum([len(vals) for vals in true_dict.values()]))
    true_dict, duplicate_cells_dict = check_neighbouring_cells(labelled_cell_jsons)  # removes duplicate/neighbouring cells
    print('all unique labelled', np.sum([len(vals) for vals in true_dict.values()]))
    valid_img_names = get_valid_images(img_names)
    # true_dict = get_true_dict_for_validation(true_dict, valid_img_names)
    nick_img_names = [x for x in valid_img_names if 'DeRuyter' in x]
    nick_dict = get_true_dict_for_validation(true_dict, nick_img_names)
    # constrain by boundary just like DL accell for direct comparison without worrying about boundary
    nick_dict_wrapper = constrain_by_ac_chamber({'all': nick_dict}, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)
    print('all valid nick labelled', np.sum([len(vals) for vals in nick_dict.values()]))
    print('all valid constrained nick labelled', np.sum([len(vals) for vals in nick_dict_wrapper['all'].values()]))

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

    # kathryn labelled
    kathryn_dict = get_true_coords(labelled_kathryn_jsons)
    print('all kathryn', np.sum([len(vals) for vals in kathryn_dict.values()]))
    kathryn_dict, kathryn_duplicate_dict = check_neighbouring_cells_from_dict(kathryn_dict)  # removes duplicate/neighbouring cells
    print('all unique kathryn', np.sum([len(vals) for vals in kathryn_dict.values()]))
    kathryn_dict_wrapper = constrain_by_ac_chamber({'all': kathryn_dict}, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)
    print('all constrained kathryn', np.sum([len(vals) for vals in kathryn_dict_wrapper['all'].values()]))

    # calculate sensitivity and false positive both ways
    # all labelled after accounting for pixel_lim
    # nick vs yue
    output_dir = os.path.join(base_folder, 'inter_obs', 'nick_vs_yue_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, nick_dict_wrapper['all'], yue_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    # reverse direction of comparison since not commutative
    output_dir = os.path.join(base_folder, 'inter_obs', 'yue_vs_nick_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, yue_dict_wrapper['all'], nick_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    save_imgs(output_dir, nick_dict_wrapper['all'], yue_dict_wrapper, img_names, raw_images, img_preds, ac_threshold=ac_threshold)

    # stricter labels for yue vs nick - take out commented stuff
    yue_dict_strict = get_review_txt_dict(review_file=yue_file, allow_commented=False)
    print('all yue strict', np.sum([len(vals) for vals in yue_dict_strict.values()]))
    yue_dict_strict, yue_duplicate_strict = check_neighbouring_cells_from_dict(yue_dict_strict)  # removes duplicate/neighbouring cells
    print('all unique yue strict', np.sum([len(vals) for vals in yue_dict_strict.values()]))
    yue_dict_strict_wrapper = constrain_by_ac_chamber({'all':yue_dict_strict}, img_names, img_preds, raw_images, converted_imgs, ac_threshold=ac_threshold)
    print('all constrained yue strict', np.sum([len(vals) for vals in yue_dict_strict_wrapper['all'].values()]))

    output_dir = os.path.join(base_folder, 'inter_obs', 'nick_vs_yue_strict_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, nick_dict_wrapper['all'], yue_dict_strict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    save_imgs(output_dir, nick_dict_wrapper['all'], yue_dict_strict_wrapper, img_names, raw_images, img_preds, ac_threshold=ac_threshold)
    output_dir = os.path.join(base_folder, 'inter_obs', 'yue_vs_nick_strict_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, yue_dict_strict_wrapper['all'], nick_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)

    # break up by class - think about relative intensity differences
    nick_dict_by_class = get_true_coords_by_class(nick_dict_wrapper['all'], imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    # yue_dict_by_class = get_true_coords_by_class(yue_dict_wrapper['all'], imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    yue_dict_strict_by_class = get_true_coords_by_class(yue_dict_strict_wrapper['all'], imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    classes = nick_dict_by_class.keys()
    output_dir = os.path.join(base_folder, 'inter_obs', 'nick_vs_yue_strict2_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    for cls in classes:
        log_sensitivity_precision(output_dir, nick_dict_by_class[cls], yue_dict_strict_by_class[cls], class_type=cls, pixel_lim=pixel_lim)
    output_dir = os.path.join(base_folder, 'inter_obs', 'yue_vs_nick_strict2_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    for cls in classes:
        log_sensitivity_precision(output_dir, yue_dict_strict_by_class[cls], nick_dict_by_class[cls], class_type=cls, pixel_lim=pixel_lim)

    ## yue vs kathryn
    output_dir = os.path.join(base_folder, 'inter_obs', 'kathryn_vs_yue_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, kathryn_dict_wrapper['all'], yue_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    # reverse direction of comparison since not commutative
    output_dir = os.path.join(base_folder, 'inter_obs', 'yue_vs_kathryn_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, yue_dict_wrapper['all'], kathryn_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    save_imgs(output_dir, kathryn_dict_wrapper['all'], yue_dict_wrapper, img_names, raw_images, img_preds, ac_threshold=ac_threshold)

    output_dir = os.path.join(base_folder, 'inter_obs', 'kathryn_vs_yue_strict_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, kathryn_dict_wrapper['all'], yue_dict_strict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    save_imgs(output_dir, kathryn_dict_wrapper['all'], yue_dict_strict_wrapper, img_names, raw_images, img_preds, ac_threshold=ac_threshold)
    output_dir = os.path.join(base_folder, 'inter_obs', 'yue_vs_kathryn_strict_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, yue_dict_strict_wrapper['all'], kathryn_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)

    # break up by class - think about relative intensity differences
    kathryn_dict_by_class = get_true_coords_by_class(kathryn_dict_wrapper['all'], imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    yue_dict_strict_by_class = get_true_coords_by_class(yue_dict_strict_wrapper['all'], imgs=raw_images, img_names=img_names)  # needs correct raw_images!
    classes = kathryn_dict_by_class.keys()
    output_dir = os.path.join(base_folder, 'inter_obs', 'kathryn_vs_yue_strict2_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    for cls in classes:
        log_sensitivity_precision(output_dir, kathryn_dict_by_class[cls], yue_dict_strict_by_class[cls], class_type=cls, pixel_lim=pixel_lim)
    output_dir = os.path.join(base_folder, 'inter_obs', 'yue_vs_kathryn_strict2_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    for cls in classes:
        log_sensitivity_precision(output_dir, yue_dict_strict_by_class[cls], kathryn_dict_by_class[cls], class_type=cls, pixel_lim=pixel_lim)

    ## nick vs kathryn
    output_dir = os.path.join(base_folder, 'inter_obs', 'kathryn_vs_nick_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, kathryn_dict_wrapper['all'], nick_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    # reverse direction of comparison since not commutative
    output_dir = os.path.join(base_folder, 'inter_obs', 'nick_vs_kathryn_ac{}_avg{}'.format(ac_threshold, int(do_avg)))
    log_sensitivity_precision(output_dir, nick_dict_wrapper['all'], kathryn_dict_wrapper['all'], class_type='total', pixel_lim=pixel_lim)
    save_imgs(output_dir, kathryn_dict_wrapper['all'], nick_dict_wrapper, img_names, raw_images, img_preds, ac_threshold=ac_threshold)
    return


if __name__ == '__main__':
    from sys import platform
    if platform == "linux" or platform == "linux2":
        base_folder = '/data/yue/pepple'
    elif platform == "win32":
        base_folder = 'z:/yue/pepple'
    orig_img_folder = os.path.join(base_folder, 'accell', 'segmentations')

    # define folder
    kathryn_folder = os.path.join(base_folder, 'accell', 'imgs_1scan_nick_kathryn')
    # img_names, imgs = get_imgs(kathryn_folder)    # visualise images
    # labelled_cells_dict = get_all_labelled_cells(img_names, imgs, visualise=False)
    #
    # # center appropriately
    # do_avg = False
    # raw_images, converted_imgs, raw_img_names, img_preds = get_img_predictions(folder=orig_img_folder)
    # aligned_img_preds = []
    # all_indices = []
    # aligned_raw_imgs = []
    # for idx, img_name in enumerate(img_names):
    #     real_idx = raw_img_names.index(img_name)
    #     all_indices.append(real_idx)
    #     aligned_img_preds.append(img_preds[idx])
    #     aligned_raw_imgs.append(raw_images[idx])    # want original without markups for recentering
    # all_cell_data, cell_labels = recenter_cells_from_dict(aligned_raw_imgs, img_names, img_preds, labelled_cells_dict, kathryn_folder, do_avg=do_avg)   # package into dict

    # compare against nick/yue
    # compare_all_users(orig_img_folder, do_avg=False, ac_threshold=.3, pixel_lim=3)
    compare_all_users(orig_img_folder, do_avg=False, ac_threshold=.9, pixel_lim=3)