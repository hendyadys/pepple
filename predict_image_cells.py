import numpy as np
import random, os, subprocess, cv2, json
from sys import platform
import time

from matplotlib import pyplot as plt
from matplotlib import patches
from scipy import ndimage
from ast import literal_eval

from data import slice_data
from analyser import combine_img, center_scale_imgs
from make_accell_data import convert_raw_imgs, convert_image, get_img_predictions, DOWNSAMPLE_RATIO, get_raw_imgs, \
    find_chamber_center, get_img_names, make_box_coords, ACCELL_DIAMETER
# from analyseCellPreds import recombine_predictions, visualise_pred_vs_truth
import data_ac_seg

patch_rows = 32
patch_cols = 32


# function for analysing new images given folder
def convert_tiff2png_accell(folder, real_load=True):
    # raw_images, img_names = get_raw_imgs(folder, ext='.tiff')
    img_names = get_img_names(folder, ext='.tiff')

    data_folder = os.path.join(folder, 'npy_data')
    converted_npy = os.path.join(data_folder, 'converted_imgs_orig_scale.npy')
    if os.path.isfile(converted_npy):
        converted_imgs = np.load(converted_npy)
        return converted_imgs, img_names
    elif not real_load:
        return np.ndarray(shape=(0, 1024, 1000)), img_names

    raw_images = np.ndarray(shape=(0, 1024, 1000))  # hack to avoid crashing
    num_images = raw_images.shape[0]  # NB - this might be different if not all 1200 slices used
    # always same size?
    converted_imgs = np.ndarray(raw_images.shape, dtype=np.float32)
    output_dir = os.path.join(folder, 'tiff2png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # convert raw images using imagemagick, read and save
    for idx, img_name in enumerate(img_names):
        converted_img_name = convert_image(folder, img_name, output_dir, options=' ')  # dont rescale image
        converted_imgs[idx, ] = cv2.imread(converted_img_name, cv2.IMREAD_GRAYSCALE)
    np.save(converted_npy, converted_imgs)
    return converted_imgs, img_names


def delete_files(folder, ext='.tiff'):
    if os.path.isdir(folder):
        ext_files = [x for x in os.listdir(folder) if ext in x.lower()]
        for f in ext_files:
            os.remove(os.path.join(folder, f))
    return 1


# call model.predict ac_seg
# call faster rcnn for predicting accells
def predict_img_seg_cells(folder, true_coord_folder=None, visualise=False, sparse=False, avg_pngs=3, use_cv_resize=True):
    start = time.time()

    # grab images from folder
    data_folder = os.path.join(folder, 'npy_data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # part 1 - predict chamber
    # convert tiffs to png at original resolution for predicting ac cells
    if use_cv_resize:
        # TODO - check code for substitute cv2.resize for imagemagick
        img_names = data_ac_seg.get_seg_raw_imgs(folder)
        raw_data, convert_imgs_orig_scale = data_ac_seg.get_raw_imgs(img_names, main_raw_folder=folder, all_raw_folder=folder, ext='TIFF', is_empty=False)
        raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder, out_folder=data_folder)
        end = time.time()
    else:
        convert_imgs_orig_scale, _ = convert_tiff2png_accell(folder)
        delete_files(os.path.join(folder, 'tiff2png'), '.png')  # converted pngs original scale
        # a) convert them from tiff
        # b) store, batch predict
        # c) reconstitute into whole images and store
        raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder, out_folder=data_folder)
        delete_files(folder, '.tiff')   # raw tiffs
        delete_files(os.path.join(folder, 'seg_scaled'), '.png')  # converted pngs 1/2 scale
        end = time.time()
        # return  # hack dont create patches on titan because less disk

    # part 2 - predict cells
    patch_imgs_folder = os.path.join(folder, 'img_patches_avg' if avg_pngs else 'img_patches')
    if not os.path.exists(patch_imgs_folder):
        os.makedirs(patch_imgs_folder)

    # for each image and its predicted chamber
    # create smallest 32*32 patches cover of predicted chamber for predicting ac cells
    half_col = int(patch_cols/2)
    half_row = int(patch_rows / 2)
    for idx, img_name in enumerate(img_names):
        # raw_image = raw_images[idx, ]
        raw_image = convert_imgs_orig_scale[idx, ]   # need to predict accells on converted png images
        raw_rows, raw_cols = raw_image.shape        # img_base = img_name.replace('.TIFF', '')
        img_base = img_name.split('.')[0]
        img_num = int(img_base.split('(')[1].split(')')[0])

        if sparse and ((img_num%avg_pngs!=0 if avg_pngs!=0 else img_num%2!=0) or (img_num < 600 or img_num > 700)):  # only every 4th of middle scans in volume
            continue
        if avg_pngs:
            raw_image = np.mean(convert_imgs_orig_scale[idx:idx+2, ], axis=0)
            # test code - even same place imaged multiple times intensities can be quite different
            # plt.figure(1)
            # plt.clf()
            # plt.subplot(221)
            # plt.imshow(convert_imgs_orig_scale[idx, ])
            # plt.subplot(222)
            # plt.imshow(convert_imgs_orig_scale[idx+1, ])
            # plt.subplot(223)
            # plt.imshow(convert_imgs_orig_scale[idx+2, ])
            # plt.subplot(224)
            # plt.imshow(np.abs(convert_imgs_orig_scale[idx, ]-raw_image))

        chamber_limits, x1, y1, x2, y2 = get_mask_boundary(img_preds[idx, ])
        # create patches based on mask on raw image (*DOWNSAMPLE_RATIO)
        # min=0, max=image_edge or chamber_edge + some margin
        for x in range(max(int(x1*DOWNSAMPLE_RATIO) - half_col, 0), min(int(x2*DOWNSAMPLE_RATIO) + patch_cols, raw_cols-patch_cols), patch_cols):
            for y in range(max(int(y1*DOWNSAMPLE_RATIO) - half_row, 0), min(int(y2*DOWNSAMPLE_RATIO) + patch_rows, raw_rows-patch_rows), patch_rows):
                cur_image = raw_image[y:y+patch_rows, x:x+patch_cols]  # careful about meaning of y(row) and x(col)
                cur_path = os.path.join(patch_imgs_folder, '{}_h{}_w{}.png'.format(img_base, y, x))
                cv2.imwrite(cur_path, cur_image)
                if cur_image.shape != (patch_rows, patch_cols):
                    print('accell img_patch creation dimension mismatch -', cur_path, cur_image.shape)

                if visualise:
                    plt.imshow(raw_image)
                    plt.scatter(x=[x1*DOWNSAMPLE_RATIO, x1*DOWNSAMPLE_RATIO, x2*DOWNSAMPLE_RATIO, x2*DOWNSAMPLE_RATIO],
                                y=[y1*DOWNSAMPLE_RATIO, y2*DOWNSAMPLE_RATIO, y1*DOWNSAMPLE_RATIO, y2*DOWNSAMPLE_RATIO],
                                c='red', s=2)
                    plt.scatter(x=[x, x, x+patch_cols, x+patch_cols], y=[y, y+patch_rows, y, y+patch_rows], c='blue', s=2)

    # # better to do this separately and call CUDA appropriately
    # # # predict ac cells on smallest 32*32 patches cover of predicted chamber - using keras
    # ac_weights_folder = os.path.join('data','yue', 'pepple', 'accell', 'ac_training3_32_32')
    # os.chdir(ac_weights_folder)
    # subprocess.call(' '.join(['CUDA_VISIBLE_DEVICES=0', '/data/yue/keras-frcnn-master/venv/Scripts/python',
    #                           '/data/yue/keras-frcnn-master/test_frcnn.py', '-p', patch_imgs_folder, '--network',
    #                           'vgg', '--config_filename', os.path.join(ac_weights_folder, 'config.pickle')]), shell=True)
    # os.chdir('/data/yue/pepple/')  # pepple

    # # check faster r-cnn results
    # results_dict = parse_predictions(patch_imgs_folder, classes=['cell', 'cell_medium', 'cell_lite'])

    # part 4 - check against truth (optional - when truth is available)
    end2 = time.time()
    # print(end - start)
    return {"seg_time":end-start, "cell_time":end2-end}


def get_mask_boundary(img_pred, pred_threshold=0.2, visualise=False):
    # pred_threshold = 0.9  # very conservative threshold
    # pred_threshold = 0.2  # less conservative for making patches for prediction
    chamber_limits = np.argwhere(img_pred > pred_threshold)  # n*2 where either 1st/2nd coord below pred_threshold
    print('chamber_limits:', chamber_limits.shape)
    if np.prod(chamber_limits.shape) > 0:
        x1 = np.min(chamber_limits[:, 1])
        x2 = np.max(chamber_limits[:, 1])
        y1 = np.min(chamber_limits[:, 0])
        y2 = np.max(chamber_limits[:, 0])
    else:
        x1=0
        x2=0
        y1=0
        y2=0

    if visualise:
        plt.imshow(img_pred)
        plt.scatter(x=[x1, x1, x2, x2], y=[y1, y2, y1, y2], c='red', s=2)
    return chamber_limits, x1, y1, x2, y2


# copied functions over from analyseBoxResults
def get_predicted_coords(folder, file_name="coords_ac.txt"):
    predicted_dict = {}
    full_path = "{}/{}".format(folder, file_name)
    if os.path.isfile(full_path):
        with open(full_path) as fin:
            for l in fin:
                arr = l.rstrip().split("\t")
                file_name = arr[0].replace('.png', '')
                if file_name in predicted_dict:
                    predicted_dict[file_name] += literal_eval(arr[1])
                else:
                    predicted_dict[file_name] = literal_eval(arr[1])
    return predicted_dict


def parse_predictions(results_folder, classes=['cell', 'cell_medium', 'cell_lite'], suffix='', true_dict=None):
    all_dict = {}
    for idx, cls in enumerate(classes):
        # class_pred_dict = get_predicted_coords(os.path.join(results_folder, 'coords_{}.txt'.format(cls)))
        class_pred_dict = get_predicted_coords(results_folder, 'coords_{}{}.txt'.format(cls, suffix))
        class_combined_pred_dict = recombine_predictions(class_pred_dict)
        all_dict[cls] = class_combined_pred_dict
        if true_dict:
            visualise_pred_vs_truth(true_dict, class_combined_pred_dict, class_type='cell')
    return all_dict


# check and visualise ac preds
def visualise_ac_preds(vol_folder, do_save=False):
    # seg preds
    pred_npy = os.path.join(vol_folder, 'npy_data', 'img_preds.npy')
    seg_preds = np.load(pred_npy)

    # raw images and sorted tiff_names
    # raw_images, tiff_names = get_raw_imgs(vol_folder, ext='.tiff')
    convert_imgs_orig_scale, tiff_names = convert_tiff2png_accell(vol_folder)

    ac_preds_folder = os.path.join(vol_folder, 'ac_preds')
    if not os.path.isdir(ac_preds_folder):
        os.makedirs(ac_preds_folder)

    for idx, tiff_name in enumerate(tiff_names):
        # visualise
        cur_img = cv2.imread(os.path.join(vol_folder, tiff_name), cv2.IMREAD_GRAYSCALE)     # by tiff_name
        # cur_img2 = raw_images[idx, ]    # these should align
        cur_img_conv = convert_imgs_orig_scale[idx,]  # these should align
        # if np.sum(cur_img==cur_img2)!=np.prod(cur_img.shape):
        #     print('image name alignment problem for {}'.format(tiff_name))
        plt.figure(1)
        plt.clf()
        plt.subplot(131)
        plt.imshow(cur_img_conv)

        plt.subplot(132)
        img_ac_preds = seg_preds[idx, ]
        plt.imshow(img_ac_preds)
        plt.title(tiff_name)

        plt.subplot(133)
        chamber_limits, mean_x, mean_y = find_chamber_center(img_ac_preds)
        scaled_img = cv2.imread(os.path.join(vol_folder, 'seg_scaled', tiff_name.replace('.TIFF', '.png')), cv2.IMREAD_GRAYSCALE)
        plt.imshow(scaled_img)
        plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='yellow', s=1)
        if do_save:
            plt.savefig(os.path.join(ac_preds_folder, tiff_name.replace('.TIFF', '.png')))
        print('{} \n mean_raw={}; std_raw={}\n mean_conv={}; std_conv={}\n mean_scaled={}; std_scaled={}'
              .format(tiff_name, np.mean(cur_img), np.std(cur_img), np.mean(cur_img_conv), np.std(cur_img_conv), np.mean(scaled_img), np.std(scaled_img) ))
    return


def range_overlap(a_min, a_max, b_min, b_max):
    return (a_min <= b_max) and (b_min <= a_max)


def visualise_volume_preds(vol_folder, conservative=0, full_vol=False, no_mid=0, ac_threshold=.9):
    plt.switch_backend('agg')
    # seg preds
    pred_npy = os.path.join(vol_folder, 'npy_data', 'img_preds.npy')
    seg_preds = np.load(pred_npy)

    # tiff_names = get_img_names(folder, ext='.tiff')
    converted_npy = os.path.join(vol_folder, 'npy_data', 'converted_imgs.npy')
    if os.path.isfile(converted_npy):
        converted_imgs = np.load(converted_npy)
    # else: return

    convert_imgs_orig_scale, tiff_names = convert_tiff2png_accell(vol_folder, real_load=False)
    if convert_imgs_orig_scale.shape[0] < 50:   # orig_scale not found - needed for plotting cells on top
        return

    # parse cell coords
    suffix = '_ac_training3_32_32'
    if platform == "linux" or platform == "linux2":
        pred_dict_cell = get_predicted_coords(os.path.join(vol_folder), 'img_patches', 'coords_cell{}.txt'.format(suffix))
        pred_dict_lite = get_predicted_coords(os.path.join(vol_folder), 'img_patches', 'coords_cell_lite{}.txt'.format(suffix))
        pred_dict_med = get_predicted_coords(os.path.join(vol_folder), 'img_patches', 'coords_cell_medium{}.txt'.format(suffix))
    else:
        if full_vol:
            pred_dict_cell = get_predicted_coords(os.path.join(vol_folder, 'full_vol'), 'coords_cell{}.txt'.format(suffix))
            pred_dict_med = get_predicted_coords(os.path.join(vol_folder, 'full_vol'), 'coords_cell_medium{}.txt'.format(suffix))
            pred_dict_lite = get_predicted_coords(os.path.join(vol_folder, 'full_vol'), 'coords_cell_lite{}.txt'.format(suffix))
        else:
            pred_dict_cell = get_predicted_coords(os.path.join(vol_folder), 'coords_cell{}.txt'.format(suffix))
            pred_dict_med = get_predicted_coords(os.path.join(vol_folder), 'coords_cell_medium{}.txt'.format(suffix))
            pred_dict_lite = get_predicted_coords(os.path.join(vol_folder), 'coords_cell_lite{}.txt'.format(suffix))

    combined_pred_dict_cell = recombine_predictions(pred_dict_cell)  # combined predicted coords
    combined_pred_dict_med = recombine_predictions(pred_dict_med)  # combined predicted coords
    combined_pred_dict_lite = recombine_predictions(pred_dict_lite)  # combined predicted coords

    # visualize on original images
    # pred_figs = os.path.join(vol_folder, 'pred_figs') if not conservative else os.path.join(vol_folder, 'pred_figs_conservative')
    pred_figs = os.path.join(vol_folder, 'pred_figs_c{}_m{}_t{}_f{}'.format(conservative, no_mid, ac_threshold, full_vol))
    if not os.path.isdir(pred_figs):
        os.makedirs(pred_figs)

    combined_preds = {}
    # tiff_names = [x for x in sorted(os.listdir(vol_folder)) if '.tiff' in x.lower()]
    # raw_images, tiff_names = get_raw_imgs(vol_folder, ext='.tiff')    # isnt actually used
    for idx, tiff_name in enumerate(tiff_names):
        img_num = int(tiff_name.split('(')[1].split(')')[0])
        if not full_vol and (img_num%2!=0 or img_num < 600 or img_num > 700):
            continue

        tiff_base = tiff_name.replace('.TIFF', '')
        combined_preds[tiff_base] = {}  # initiate

        # visualise
        # cur_img = cv2.imread(os.path.join(vol_folder, tiff_name), cv2.IMREAD_GRAYSCALE)   # if tiffs are there
        # cur_img = raw_images[idx, ]
        cur_img = convert_imgs_orig_scale[idx, ]
        # cur_img = converted_imgs[idx,]
        img_ac_preds = seg_preds[idx, ]
        mid_limits, mid_min, mid_max = any_middle_stripes(cur_img, avg_period=10, intensity_threshold=180)

        # # raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder)
        # # plt.switch_backend('TkAgg')
        # plt.figure(3)   # visualize preds for cur_img
        # plt.clf()
        # plt.subplot(121)
        # plt.imshow(converted_imgs[idx, ])
        # plt.subplot(122)
        # plt.imshow(converted_imgs[idx, ])
        # chamber_limits, chamber_size, mean_x, mean_y = calc_img_chamber_size(img_ac_preds, pred_threshold=.8)
        # plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='yellow', s=1)

        # chamber_limits_no_center = []
        # for seg_coord in chamber_limits:
        #     seg_y, seg_x = seg_coord
        #     if seg_x*DOWNSAMPLE_RATIO not in mid_limits:
        #         chamber_limits_no_center.append(seg_coord)
        # chamber_limits_no_center = np.asarray(chamber_limits_no_center)
        # if mid_min is not None and mid_max is not None:
        #     chamber_limits_no_center = chamber_limits[(chamber_limits[:, 1]*DOWNSAMPLE_RATIO<mid_min) | (chamber_limits[:, 1]*DOWNSAMPLE_RATIO>mid_max), ]
        # else:
        #     chamber_limits_no_center = chamber_limits
        # plt.scatter(x=chamber_limits_no_center[:, 1], y=chamber_limits_no_center[:, 0], c='yellow', s=1)
        # plt.savefig(os.path.join(pred_figs, '{}_chamber_seg.png'.format(tiff_base)), bbox_inches='tight')

        fig1, ax1 = plt.subplots(1)
        ax1.imshow(cur_img)
        chamber_limits, chamber_size, mean_x, mean_y = calc_img_chamber_size(img_ac_preds, pred_threshold=ac_threshold)
        combined_preds[tiff_base]['chamber_size'] = chamber_size

        if tiff_base in combined_pred_dict_cell:
            tiff_cells = combined_pred_dict_cell[tiff_base]
            # combined_preds[tiff_base]['cell'] = tiff_cells
            combined_preds[tiff_base]['cell'] = []
            for tiff_cell in tiff_cells:
                if is_cell_in_ac(tiff_cell, chamber_limits, mean_x, mean_y, conservative=conservative, img=cur_img,
                                 no_mid=no_mid, mid_min=mid_min, mid_max=mid_max):
                    combined_preds[tiff_base]['cell'].append(tiff_cell)
                    plot_cell(ax1, tiff_cell, color='white')

        if tiff_base in combined_pred_dict_med:
            tiff_cells = combined_pred_dict_med[tiff_base]
            combined_preds[tiff_base]['cell_medium'] = []
            for tiff_cell in tiff_cells:
                if is_cell_in_ac(tiff_cell, chamber_limits, mean_x, mean_y, conservative=conservative, img=cur_img,
                                 no_mid=no_mid, mid_min=mid_min, mid_max=mid_max):
                    combined_preds[tiff_base]['cell_medium'].append(tiff_cell)
                    plot_cell(ax1, tiff_cell, color='yellow')

        if tiff_base in combined_pred_dict_lite:
            tiff_cells = combined_pred_dict_lite[tiff_base]
            combined_preds[tiff_base]['cell_lite'] = []
            # dont run to save computation cost/speed
            # for tiff_cell in tiff_cells:
            #     if is_cell_in_ac(tiff_cell, chamber_limits, mean_x, mean_y, conservative=conservative, img=None):
            #         combined_preds[tiff_base]['cell_lite'].append(tiff_cell)

        # save plot
        plt.savefig(os.path.join(pred_figs, '{}.png'.format(tiff_base)), bbox_inches='tight')

    # save combined_preds dict
    json_path = os.path.join(vol_folder, 'chamber_ac_preds.json')
    with open(json_path , 'w') as fout:  # save for easy access
        json.dump(combined_preds, fout)
        # json.dump(combined_preds['20171009mouse4_Day1_Right (600)'], fout)
    fout.close()

    # process volume preds and save
    volume_preds = '{}_preds_c{}_m{}_t{}.csv'.format(vol_folder.split('\\')[-1], conservative, no_mid, ac_threshold)
    # if conservative:
    #     volume_preds = '{}{}'.format(volume_preds, '_conservative')
    # if full_vol:
    #     volume_preds = '{}{}'.format(volume_preds, '_full_vol')
    # volume_preds = '{}.csv'.format(volume_preds)
    with open(os.path.join(vol_folder, volume_preds), 'w') as fout:
        # header line
        fout.write('img, chamber_size, #cell, area_cell, #cell_medium, area_cell_medium, #cell_lite, area_cell_lite\n')
        for fname, fdata in combined_preds.items():
            if 'cell' in fdata:
                num_cells, area_cell = get_cell_stats(fdata['cell'])
            else:
                num_cells, area_cell = 0, 0
            if 'cell_medium' in fdata:
                num_med, area_med = get_cell_stats(fdata['cell_medium'])
            else:
                num_med, area_med = 0, 0
            if 'cell_lite' in fdata:
                num_lite, area_lite = get_cell_stats(fdata['cell_lite'])
            else:
                num_lite, area_lite = 0, 0
            img_info = [fname, ff(fdata['chamber_size'] * (DOWNSAMPLE_RATIO ** 2)), ff(num_cells), ff(area_cell), ff(num_med), ff(area_med), ff(num_lite), ff(area_lite)]
            fout.write('{} \n'.format(','.join(img_info)))
    fout.close()
    return combined_preds


def get_cell_stats(cell_coords):
    num_cells = len(cell_coords)
    area_cells = []
    for cell_coord in cell_coords:
        x1, y1, x2, y2 = cell_coord
        area_cells.append((x2-x1)*(y2-y1))
    return num_cells, np.sum(area_cells)


# int formatter - since integer pixel areas and counts
def ff(mynum):
    # return '{0:.2f}'.format(mynum)
    return '{0:d}'.format(int(round(mynum)))


def calc_img_chamber_size(mask, img=None, pred_threshold=.9):
    # pred_threshold = 0.9    # very conservative threshold
    chamber_size = np.sum(mask > pred_threshold)
    chamber_limits = np.argwhere(mask > pred_threshold)    # n*2 where either 1st/2nd coord below pred_threshold
    mean_x, mean_y = get_chamber_center(chamber_limits)

    if img is not None:  # visualise
        plt.figure(1)
        plt.imshow(img[::2, ::2])   # every other pixel downsampling for visual purposes
        plt.scatter(x=chamber_limits[:, 1], y=chamber_limits[:, 0], c='yellow', s=1)
        plt.scatter(x=mean_x, y=mean_y, c='green', s=1)

    return chamber_limits, int(chamber_size), mean_x, mean_y


def is_cell_in_ac(tiff_cell, chamber_limits, mean_x, mean_y, conservative=0, no_mid=0, mid_min=None, mid_max=None,
                  img=None, visualise=False):
    if len(tiff_cell)==4:
        x1, y1, x2, y2 = tiff_cell  # coords on 1024*1000
    elif len(tiff_cell)==5:
        x1, y1, x2, y2, prob = tiff_cell  # coords on 1024*1000
    elif len(tiff_cell)==2:
        x1, y1 = tiff_cell
        # x2, y2 = x1, y1  # repeat for ease of code flow
        x1, y1, x2, y2 = make_box_coords((x1, y1), img.shape, box_size=ACCELL_DIAMETER)

    no_mid_stripe = True
    if no_mid and mid_min is not None and mid_max is not None:  # not in middle stripe -> True
        # no_mid_stripe = not range_overlap(x1, x2, mid_min, mid_max)
        no_mid_stripe = not range_overlap(x1, x2, mid_min-conservative, mid_max+conservative)
        # no_mid_stripe = True if (x1 < mid_min or x1 > mid_max) and (x2 < mid_min or x2 > mid_max) else False

    top_in_ac = in_chamber((x1,y1), chamber_limits, mean_x, mean_y, conservative=conservative, img=img)
    bottom_in_ac = in_chamber((x2,y2), chamber_limits, mean_x, mean_y, conservative=conservative, img=img)
    return no_mid_stripe and top_in_ac and bottom_in_ac


# check middle band
def any_middle_stripes(img, avg_period=10, intensity_threshold=200, visualise=False, out_base=None, out_num=None):
    img_rows, img_cols = img.shape
    avg_intensities = np.zeros(shape=(img_cols, 1))
    for idx in range(img_cols):     # traverse by x
        avg_intensities[idx] = np.mean(img[0, idx:idx+avg_period])  # take top row for now
    middle_limits = np.argwhere(avg_intensities > intensity_threshold)  #
    # if len(middle_limits)>0:
    middle_limits = middle_limits[:, 0]     # just need column indices
    # middle_limits = middle_limits[:, 0] * DOWNSAMPLE_RATIO  # just need column indices

    # # vs avg intensity
    # img_mean = np.mean(img)
    # middle_limits2 = np.argwhere(img[0, ] > img_mean*7)
    # print('middle_limits:', middle_limits.shape, 'middle_limits2', middle_limits2.shape)

    # only central parts - edges also flare sometimes
    central_low = 450
    central_high = 550
    if len(middle_limits) > 0:
        middle_limits = middle_limits[np.where((middle_limits>central_low) & (middle_limits<central_high))]   # for 1000 cols
    # if len(middle_limits2) > 0:
    #     middle_limits2 = middle_limits2[np.where((middle_limits2>central_low) & (middle_limits2<central_high))]  # for 1000 cols

    if visualise:
        plt.figure(1)
        plt.clf()
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(img)
        for x in middle_limits:
            plt.scatter(x=x*np.ones(shape=(img_rows, 1)), y=np.asarray(range(0, img_rows)), c='yellow', s=1)
        # plt.subplot(133)
        # plt.imshow(img)
        # for x in middle_limits2:
        #     plt.scatter(x=x*np.ones(shape=(img_rows, 1)), y=range(0, img_rows), c='yellow', s=1)

        if out_base:
            outfolder = os.path.join(out_base, 'middle_stripe')
            if not os.path.isdir(outfolder):
                os.makedirs(outfolder)
            plt.savefig(os.path.join(outfolder, '{}.png'.format(out_num)))

    middle_min = np.min(middle_limits) if len(middle_limits) > 0 else None
    middle_max = np.max(middle_limits) if len(middle_limits) > 0 else None
    return middle_limits, middle_min, middle_max


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


def plot_cell(ax1, cell, color='red'):
    x1, y1, x2, y2 = cell
    ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1))
    return


def copy_central_slices(vol_folder, mid_low=600, mid_high=700, ext='.tiff'):
    from shutil import copyfile, copy2

    out_folder = os.path.join('volume_data2', vol_folder.replace('D:', ''))
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    img_names = [x for x in sorted(os.listdir(vol_folder)) if ext in x.lower() and 'mask' not in x.lower()]
    for idx, img_name in enumerate(img_names):
        img_num = int(img_name.split('(')[1].split(')')[0])
        if vol_folder in ['D:20170920mouse5_Day2 Right']:   # mislabelled - will cause problems in predict_img_seg_cells()
            img_num -= 1200

        if img_num%2!=0 or (img_num < mid_low or img_num > mid_high):
            continue

        src_path = os.path.join(vol_folder, img_name)
        # copyfile(src_path, os.path.join(out_folder, img_name))
        copy2(src_path, out_folder)
    return


def check_training_data(train_folder, ext='.png'):
    img_names = [x for x in os.listdir(train_folder) if ext in x.lower()]
    num_imgs = len(img_names)
    img_np = np.ndarray((num_imgs, patch_rows, patch_cols))

    for idx, img_name in enumerate(img_names):
        temp = cv2.imread(os.path.join(train_folder, img_name), cv2.IMREAD_GRAYSCALE)
        if temp is None or np.isnan(np.mean(temp)):
            print('nan values in {}'.format(os.path.join(train_folder, img_name)))
        img_np[idx, ] = temp
    print('img stats for {}\n mean={}; std={}'.format(train_folder, np.mean(img_np), np.std(img_np)))
    return img_np


def check_img_patches(folder, ext='.png'):
    patch_path = os.path.join(folder, 'img_patches')
    patch_names = [x for x in os.listdir(patch_path) if ext in x.lower()]

    convert_imgs_orig_scale, img_names = convert_tiff2png_accell(folder)
    data_folder = os.path.join(folder, 'npy_data')
    raw_images, converted_imgs, img_names, img_preds = get_img_predictions(folder, out_folder=data_folder)

    sep = '_'
    for idx, patch_name in enumerate(patch_names):
        patch_data = cv2.imread(os.path.join(patch_path, patch_name), cv2.IMREAD_GRAYSCALE)
        patch_name_toks = patch_name.split(sep)
        # '20170703mouse3_Day1_Left (648)_h356_w512'
        patch_y = int(patch_name_toks[-2].replace('h', ''))
        patch_x = int(patch_name_toks[-1].replace('.png', '').replace('w', ''))

        patch_img_base = sep.join(patch_name_toks[:-2])
        patch_img_name = '{}.TIFF'.format(patch_img_base)
        img_idx = img_names.index(patch_img_name)

        raw_img = raw_images[img_idx, ]
        conv_img = convert_imgs_orig_scale[img_idx, ]
        plt.figure(1)
        plt.imshow(raw_img)
        plt.figure(2)
        plt.imshow(conv_img)
        print(np.mean(raw_img), np.std(raw_img), np.mean(conv_img), np.std(conv_img))
        # patch value comparisons vs patches from raw_img and conv_img
        raw_patch = raw_img[patch_y:patch_y+patch_rows, patch_x:patch_x+patch_cols]
        conv_patch = conv_img[patch_y:patch_y+patch_rows, patch_x:patch_x+patch_cols]
        patch_correct = np.sum(conv_patch == patch_data)
        print(np.mean(raw_patch), np.std(raw_patch), np.mean(conv_patch), np.std(conv_patch), np.mean(patch_data),
              np.std(patch_data), np.sum(raw_patch == patch_data), patch_correct)
        if patch_correct!=np.prod(patch_data.shape):
            print('{} formed incorrectly. {}'.format(patch_name, patch_correct))
    return


def remove_npy(base_folder):
    # left and Day-7 are controls
    folders = [x for x in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, x)) and ('Left' in x or 'Day-7' in x)]
    for idx, folder in enumerate(folders):
        os.remove(os.path.join(base_folder, folder, 'npy_data', 'raw_images.npy'))
        os.remove(os.path.join(base_folder, folder, 'npy_data', 'converted_imgs_orig_scale.npy'))
    return 1


def make_movie_images(vol_folder):
    # plt.switch_backend('agg')
    pred_npy = os.path.join(vol_folder, 'npy_data', 'img_preds.npy')
    seg_preds = np.load(pred_npy)

    # tiff_names = get_img_names(folder, ext='.tiff')
    converted_npy = os.path.join(vol_folder, 'npy_data', 'converted_imgs.npy')
    if os.path.isfile(converted_npy):
        converted_imgs = np.load(converted_npy)
    # else: return

    convert_imgs_orig_scale, tiff_names = convert_tiff2png_accell(vol_folder, real_load=False)
    if convert_imgs_orig_scale.shape[0] < 50:  # orig_scale not found - needed for plotting cells on top
        return

    json_path = os.path.join(vol_folder, 'chamber_ac_preds.json')
    fin = open(json_path ).read()
    vol_ac_dict = json.loads(fin)

    ac_threshold = .9
    movie_folder = os.path.join(vol_folder, 'movies_t{}'.format(ac_threshold))
    if not os.path.isdir(movie_folder):
        os.makedirs(movie_folder)

    for idx, tiff_name in enumerate(tiff_names):
        experiment_name = tiff_name.split()[0]
        tiff_base = tiff_name.replace('.TIFF', '')
        img_num = int(tiff_base .split('(')[1].split(')')[0])
        if img_num % 5 != 0:  # for speed reasons
            continue
        if img_num > 350 and img_num < 900:
            ac_threshold = .2

        # cur_img = convert_imgs_orig_scale[idx, ]
        cur_img = converted_imgs[idx, ]
        chamber_preds = seg_preds[idx, ]

        plt.figure(1)
        plt.clf()
        plt.subplot(131)
        plt.imshow(cur_img)
        plt.axis('off')

        img_ac_preds = seg_preds[idx, ]
        mid_limits, mid_min, mid_max = any_middle_stripes(cur_img, avg_period=10, intensity_threshold=180)
        chamber_limits, chamber_size, mean_x, mean_y = calc_img_chamber_size(chamber_preds, pred_threshold=ac_threshold)

        if mid_min is not None and mid_max is not None: # dont scale since plotting on scaled down images for consistency
            chamber_limits_no_center = chamber_limits[(chamber_limits[:, 1] < mid_min) |
                                                      (chamber_limits[:, 1] > mid_max), ]
        else:
            chamber_limits_no_center = chamber_limits
        plt.subplot(132)
        plt.imshow(cur_img)
        plt.axis('off')
        plt.scatter(x=chamber_limits_no_center[:, 1], y=chamber_limits_no_center[:, 0], c='yellow', s=1)

        plt.subplot(133)
        plt.imshow(cur_img)
        plt.axis('off')

        tiff_pred_dict = vol_ac_dict[tiff_base]
        for cname, c_cells in tiff_pred_dict.items():
            if cname == 'chamber_size':
                continue
            for class_cell in c_cells:
                x1, y1, x2, y2 = class_cell
                color = 'red'  # more obvious

                x = (x1 + x2) / 2.
                y = (y1 + y2) / 2.
                orig_scale = False
                if not orig_scale:
                    x = x / DOWNSAMPLE_RATIO
                    y = y / DOWNSAMPLE_RATIO
                plt.scatter(x=x, y=y, c=color, s=2)

        out_img_name = '{}_{:04}.png'.format(experiment_name, img_num)
        plt.savefig(os.path.join(movie_folder, out_img_name), bbox_inches='tight')
        # cv2.imwrite(patch_path, cur_patch)

    return


def make_movie(image_folder):
    video_name = os.path.join(image_folder, '{}.avi'.format('pred_video'))

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    # images = sorted(images, key=lambda img: int(img.split('_')[-1].replace('.png', '').replace('i', '')))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fps =2
    video = cv2.VideoWriter(video_name, -1, fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    return


if __name__ == '__main__':
    # # make movie
    # folder_path = os.path.join('volume_data', 'vol_full')
    # folders = [x for x in sorted(os.listdir(os.path.join(folder_path)))]
    # for folder in folders:
    #     if 'vol_full' in folder_path:
    #         if 'Left' in folder or 'Day-7' in folder:
    #             continue
    #         elif 'mouse2' in folder or 'mouse3' in folder or 'mouse4' in folder:
    #             continue
    #     make_movie(os.path.join(folder_path, folder, 'movies_t0.9_fixed'))
    #
    # # make movie images
    # folder_path = os.path.join('volume_data', 'vol_full')
    # folders = [x for x in sorted(os.listdir(os.path.join(folder_path)))]
    # for folder in folders:
    #     if 'vol_full' in folder_path:
    #         if 'Left' in folder or 'Day-7' in folder:
    #             continue
    #         elif 'mouse2' in folder or 'mouse3' in folder or 'mouse4' in folder:
    #             continue
    #     make_movie_images(os.path.join(folder_path, folder))

    # # test for middle images
    # folder = '20170703mouse2_Day1_Right'
    # folders = ['20171009mouse6_Day-7 Right', '20170717mouse10_Day2_Right', '20171011mouse8_Day1_Right',
    #            '20170717mouse1_Day1_Right', '20170703mouse9_Day1_Right']
    # for folder in folders:
    #     folder_path = os.path.join('volume_data', 'vol_to_analyse', folder)
    #     converted_npy = os.path.join(os.path.join(folder_path, 'npy_data'), 'converted_imgs_orig_scale.npy')
    #     converted_imgs = np.load(converted_npy)
    #     num_imgs, img_rows, img_cols = converted_imgs.shape
    #     for idx in range(num_imgs):
    #         img = converted_imgs [idx,]
    #         any_middle_stripes(img, visualise=True, out_base=folder_path, out_num=idx)
            # any_middle_stripes(img, )

    # remove_npy(os.path.join('volume_data', 'vol_to_run'))
    # remove_npy(os.path.join('volume_data', 'volume_done'))

    # # copy over mid slices for ac seg and then accell prediction purposes
    # base_path = 'D:'
    # vol_folders = [x for x in sorted(os.listdir(base_path)) if os.path.isdir(os.path.join(base_path, x)) and '2017' in x]
    # # vol_folders = [os.path.join(base_path, '20170920mouse5_Day2 Right')]
    # for idx, vol_folder in enumerate(vol_folders):
    #     if idx > 200:  # enough files
    #         break
    #     # if idx < 295:    # already processed
    #     #     continue
    #     # if 'Left' in vol_folder or 'Day-7' in vol_folder:   # these are controls -> skip
    #     #     continue
    #     if not ('Right' in vol_folder and 'Day-7' in vol_folder):   # these are controls -> skip
    #         continue
    #     print(idx, vol_folder)
    #     copy_central_slices(os.path.join(base_path, vol_folder))

    # test delete files
    # delete_files(os.path.join('volume_data', '20170703mouse3_Day1_Left'), '.tiff')

    # # predict_img_seg_cells(folder='./accell/segmentations')
    base_folder = './volume_data/vol_to_analyse'
    volume_folders = [name for name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, name))]
    elapsed_time = []
    for idx, vol_folder in enumerate(volume_folders):
        if idx > 4: break  # HACK
        folder_time = predict_img_seg_cells(folder=os.path.join(base_folder, vol_folder), sparse=True)
        elapsed_time.append(folder_time)

    fname = 'pred_times.txt'
    with open(fname, 'w') as fout:
        json.dump(elapsed_time, fout)
    fout.close()

    ## check convert (imagemagick) worked by looking at summary stats
    # check_training_data(train_folder=os.path.join('accell', 'ac_training3_32_32', 'train'))
    # check_training_data(train_folder=os.path.join('accell', 'ac_training3_32_32', 'valid'))
    # check_training_data(train_folder=os.path.join('volume_data', '20170703mouse3_Day1_Left'))
    # check_training_data(train_folder=os.path.join('volume_data', '20170703mouse3_Day1_Left', 'seg_scaled'))
    # check_training_data(train_folder=os.path.join('volume_data', '20170703mouse3_Day1_Left', 'tiff2png'))
    # check_training_data(train_folder=os.path.join('volume_data', '20170703mouse3_Day1_Left', 'img_patches'))

    # # check acseg for converted and scaled png form raw tiffs by visualisation and saving plots
    # visualise_ac_preds(os.path.join('volume_data', '20170703mouse3_Day1_Left'), do_save=True)

    # # check accell preds on patches
    folder_path = os.path.join('volume_data', 'vol_full')
    folder_path = os.path.join('volume_data', 'vol_to_analyse')
    folders = [x for x in sorted(os.listdir(os.path.join(folder_path))) if x != 'volume_done']
    # folders = ['20171009mouse4_Day1_Right']
    for folder in folders:
        if 'vol_full' in folder_path:
            if 'Left' in folder or 'Day-7' in folder:
                continue
            elif 'mouse2' in folder or 'mouse3' in folder or 'mouse4' in folder:
                continue
        start = time.time()
        # 5 pixels in, # in middle, # high threshold for ac chamber
        full_vol = True if 'vol_full' in folder_path else False
        visualise_volume_preds(os.path.join(folder_path, folder), conservative=5, no_mid=1, ac_threshold=.95, full_vol=full_vol)
        end = time.time()
        print('{}; start={}; end={}; time={}'.format(folder, start, end, (end-start)/60))


    folders = [x for x in os.listdir(os.path.join('volume_data', 'vol_full')) if x != 'volume_done']
    # for folder in folders:
    #     start = time.time()
    #     visualise_volume_preds(os.path.join('volume_data', 'vol_full', folder), conservative=False, full_vol=3)
    #     visualise_volume_preds(os.path.join('volume_data', 'vol_full', folder), conservative=True, full_vol=3)
    #     end = time.time()
    #     print('{}; start={}; end={}; time={}'.format(folder, start, end, (end-start)/60))

    # # check patch created from segmented chambers
    # check_img_patches(os.path.join('volume_data', '20170703mouse3_Day1_Left'))