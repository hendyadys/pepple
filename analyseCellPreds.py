import numpy as np
import glob, cv2, os, json
from ast import literal_eval

import matplotlib.pyplot as plt
from matplotlib import patches
# plt.switch_backend('agg')
# from make_bbox_data import downsample, DOWNSAMPLE_RATIO, get_coords

from sys import platform
if platform == "linux" or platform == "linux2":
    base_folder = '/home/yue/pepple/accell'
    orig_img_folder = os.path.join(base_folder, 'segmentations')
    empty_img_folder = os.path.join(base_folder, 'empty_segmentations')
    json_folder = os.path.join(base_folder, 'jsons')

    # # vertical transform
    # results_base = '/home/yue/pepple/runs/2017-08-09-10-20-24'
    # test_weights = 'weights-improvement-050--0.95407502.hdf5'

    # with empty seg training data
    results_base = '/home/yue/pepple/runs/2017-11-09-10-26-19'
    # test_weights = 'weights-improvement-025--0.95883078.hdf5'
    # test_weights = 'weights-improvement-050--0.95777710.hdf5'
    test_weights = 'weights-improvement-100--0.96128662.hdf5'

    # # re-ran vertical transform (lr=1e-5)
    # results_base = '/home/yue/pepple/runs/2017-11-09-23-47-29'
    # test_weights = 'weights-improvement-100--0.96851523.hdf5'
elif platform == "win32":
    base_folder = './accell'
    orig_img_folder = os.path.join(base_folder, 'segmentations')
    empty_img_folder = os.path.join(base_folder, 'empty_segmentations')
    json_folder = os.path.join(base_folder, 'jsons')

    results_base = './runs/runEmptySeg'
    test_weights = 'weights-improvement-100--0.96128662.hdf5'

    # # re-ran vertical transform (lr=1e-5)
    # results_base = './runs/runVerticalNew'
    # test_weights = 'weights-improvement-100--0.96851523.hdf5'

results_figs = os.path.join(results_base, 'figs')
results_folder = os.path.join(results_base, 'weights')

img_rows = 1024  # height
img_cols = 1000  # width


def make_image_patches(img, img_name, output_folder, remove_chamber=False, save_img=True, visualise=False):
    if not visualise:
        plt.switch_backend('agg')

    zero_pad = np.zeros((img_rows, img_rows - img_cols), dtype=np.uint8)
    img = np.concatenate((img, zero_pad), axis=1)   # right-edge (width) padding
    image_rows, image_cols = img.shape  # should be 1024*1000

    patch_size = 32
    patches_per_row = int(np.ceil(image_rows/float(patch_size)))
    patches_per_col = int(np.ceil(image_cols/float(patch_size)))
    img_patches = np.ndarray((patches_per_row, patches_per_col, patch_size, patch_size), dtype=np.float32)

    for h in range(0, image_rows, patch_size):
        for w in range(0, image_cols, patch_size):
            cur_patch = img[h:h+patch_size, w:w+patch_size]
            img_patches[int(h/patch_size), int(w/patch_size), ] = cur_patch

            if visualise:
                plt.figure(1)
                plt.imshow(cur_patch)
                plt.figure(2)
                plt.clf()
                plt.imshow(img)
                plt.scatter(x=[w, w+patch_size], y=[h, h+patch_size], c='red', s=2)
                1

            # save the patches prediction purposes
            if save_img:
                patch_name = '{}_h{}_w{}.png'.format(img_name.replace('.png', ''), h, w)
                patch_path = os.path.join(output_folder, patch_name)
                cv2.imwrite(patch_path, cur_patch)
                # plt.imshow(cur_patch)
                # plt.savefig(patch_path)
    return img_patches


def create_pred_img_patches(is_traced=True):
    ac_imgs, img_names = get_traced_images(is_traced)
    output_fname = 'seg_accell_img_patches' if is_traced else 'empty_accell_img_patches'
    output_folder = os.path.join(base_folder, output_fname)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_images = len(img_names)
    all_patches = np.ndarray((num_images, 32, 32, 32, 32), dtype=np.float32)
    for idx, img_name in enumerate(img_names):
        img_patches = make_image_patches(ac_imgs[idx, ], img_name, output_folder, remove_chamber=False, save_img=False, visualise=True)
        all_patches[idx, ] = img_patches

    np.save('{}/img_patches.npy'.format(output_folder), all_patches)
    # save img names
    with open('{}/img_names.txt'.format(output_folder), 'w') as fout:
        for img_name in img_names:
            fout.write('{}\n'.format(img_name))
    return all_patches


def load_patch_npy(is_traced=True):
    output_fname = 'seg_accell_img_patches' if is_traced else 'empty_accell_img_patches'
    npy_file = os.path.join(base_folder, output_fname, 'img_patches.npy')
    all_patch_data = np.load(npy_file)
    img_names = []
    img_file = os.path.join(base_folder, output_fname, 'img_names.txt')
    with open(img_file, 'r') as fin:
        for l in fin:
            img_names.append(l.rstrip())
    return all_patch_data, img_names


def get_ac_preds_for_images(is_traced=True):
    from train import get_unet
    from analyser import predict_image_mask

    ac_imgs, img_names = get_traced_images(is_traced)

    pred_npy = os.path.join(base_folder, 'traced_ac_mask_preds.npy' if is_traced else 'empty_ac_mask_preds.npy')
    if os.path.isfile(pred_npy):
        pred_data = np.load(pred_npy)
        return pred_data

    # otherwise predict segmentations
    model = get_unet()
    weight_path = '{}/{}'.format(results_folder, test_weights)
    model.load_weights(weight_path)

    data_shape = ac_imgs.shape
    num_images = data_shape[0]
    pred_data = np.ndarray(data_shape, dtype=np.float32)
    for idx in range(num_images):
        cur_img = ac_imgs[idx, ]
        mask_pred, strip_preds = predict_image_mask(model, cur_img)  # slice, predict and re-constitute
        pred_data[idx, ] = mask_pred
        # np.save('{}/{}_preds.npy'.format(base_folder, img_names[idx]), strip_preds)
    np.save(pred_npy, pred_data)

    return ac_imgs, img_names, pred_data


def get_traced_images(is_traced=True):
    target_folder = orig_img_folder if is_traced else empty_img_folder
    if is_traced:
        save_path = os.path.join(base_folder, 'traced_ac_images.npy')
    else:
        save_path = os.path.join(base_folder, 'empty_ac_images.npy')

    images = sorted(os.listdir(target_folder))
    real_images = [x for x in images if 'mask' not in x and '.png' in x]
    if os.path.isfile(save_path):
        img_npy = np.load(save_path)
        return img_npy, real_images

    total = len(real_images)
    img_npy = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    counter = 0
    for image in real_images:
        cur_image = cv2.imread(os.path.join(target_folder, image), cv2.IMREAD_GRAYSCALE)
        img_npy[counter, ] = cur_image
        counter +=1

    np.save(save_path, img_npy)
    return img_npy, real_images


def num_cells_within_seg():
    return


def visualize_all_preds(all_patch_data, img_names, pred_dict, true_dict):
    patch_size = 32
    for key, value in sorted(pred_dict.items()):
        key_toks = key.split('_')
        img_key = '_'.join(key_toks[:-2])
        h_adj = int(key_toks[-2].replace('h', ''))
        w_adj = int(key_toks[-1].replace('w', ''))
        img_base = img_key+'.png'
        img_idx = img_names.index(img_base)
        # img = all_patch_data[img_idx, :]
        img = cv2.imread('{}/{}'.format(orig_img_folder, img_base), cv2.IMREAD_GRAYSCALE)
        cur_patch = all_patch_data[img_idx, int(h_adj/patch_size), int(w_adj/patch_size), ]
        cur_patch2 = img[h_adj:h_adj+patch_size, w_adj:w_adj+patch_size]
        np.sum(cur_patch==cur_patch2)   # should be patch_size*patch_size

        # plt.figure(1)
        # plt.imshow(img)
        # plt.title('current img')
        #
        # plt.figure(2)
        # plt.imshow(cur_patch)
        # plt.title('current patch in img')

        pred_coords = value
        true_coords = true_dict[img_key]

        fig1, ax1 = plt.subplots(1)
        ax1.imshow(cur_patch)
        for coord in pred_coords:  # add cells to image
            (x1, y1, x2, y2) = coord
            ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=1))

        true_coords_in_patch = []
        for coord in true_coords:  # add cells to image
            (x, y) = coord
            if (x >= w_adj and x < w_adj+32) and (y >= h_adj and y < h_adj+32):
                true_coords_in_patch.append(coord)
                ax1.scatter(x=[x - w_adj], y=[y - h_adj], c='white', s=2)

        fig2, ax2 = plt.subplots(1)
        ax2.imshow(img)
        ax2.add_patch(patches.Rectangle((w_adj, h_adj), patch_size, patch_size, fill=False, color='yellow', linewidth=1))
        for coord in pred_coords:  # add cells to image
            (x1, y1, x2, y2) = coord
            ax2.add_patch(patches.Rectangle((x1+w_adj, y1+h_adj), x2 - x1, y2 - y1, fill=False, color='red', linewidth=1))

        for coord in true_coords:  # add cells to image
            (x, y) = coord
            ax2.scatter(x=[x], y=[y], c='white', s=2)
        print(key, img_key, len(pred_coords), len(true_coords_in_patch))

    return


def combined_patch_preds(is_traced=True):
    # get patch data for plotting purposes
    all_patch_data, img_names = load_patch_npy(is_traced)

    # read coords file into dictionary
    true_dict = get_true_coords(json_folder)
    # parse coords file
    pred_dict = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'))
    pred_dict = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'))
    # visualize_all_preds(all_patch_data, img_names, pred_dict, true_dict)

    # combined_pred_dict = recombine_predictions(pred_dict, all_patch_data, img_names)    # combined predicted coords
    combined_pred_dict = recombine_predictions(pred_dict)  # combined predicted coords
    visualise_pred_vs_truth(true_dict, combined_pred_dict)

    # pred closeness based on predicted and actual
    # precision/sensitivity measures overall
    # precision/sensitivity measures within seg chamber
    return


def recombine_predictions(pred_dict, patch_data=None, img_names=None):
    visualise = False
    # if patch_data is not None or img_names is not None:
    #     visualise=True

    combined_pred_dict = {}
    for key, value in pred_dict.items():
        key_toks = key.split('_')
        img_key = '_'.join(key_toks[:-2])
        h_adj = int(key_toks[-2].replace('h', ''))
        w_adj = int(key_toks[-1].replace('w', ''))

        adj_value = []
        for coord in value:
            x1, y1, x2, y2 = coord
            adj_value.append((x1+w_adj, y1+h_adj, x2+w_adj, y2+h_adj))
            # adj_value.append((x1+h_adj, y1+w_adj, x2+h_adj, y2+w_adj))

        if img_key in combined_pred_dict:
            combined_pred_dict[img_key] += adj_value
        else:
            combined_pred_dict[img_key] = adj_value

        if visualise:
            cur_idx = img_names.index(img_key + '.png')
            cur_patch = patch_data[cur_idx, int(h_adj/32), int(w_adj/32), ]
            # visualise_on_patch(key, cur_patch, h_adj, w_adj)
            print(patch_stats(cur_patch))
            plot_img_boxes(key, cur_patch, value)
    return combined_pred_dict


def visualise_pred_vs_truth(true_dict, combined_pred_dict, class_type='cell'):
    for key, value in combined_pred_dict.items():
        true_data = true_dict[key]
        img = plot_img_coords(orig_img_folder, key, true_data)
        # plot_img_boxes(key, img, value, np.asarray(true_data))
        # plot_img_boxes(key, img, value, None)
        plot_img_boxes(key, img, value, np.asarray(true_data))
        plt.title('{} predictions for {}'.format(class_type, key))
    return


# multi-class on same plot
def visualise_preds_multi_class(true_dict, combined_pred_dicts, class_names):
    colors = ['white', 'yellow', 'pink']
    file_names = combined_pred_dicts[0].keys()

    for fname in file_names:
        true_coords = np.asarray(true_dict[fname])
        fpath = os.path.join(orig_img_folder, fname + '.png')
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(img)
        ax1.scatter(x=true_coords[:,0], y=true_coords[:, 1], c='red', s=2)

        for idx, combined_pred_dict in enumerate(combined_pred_dicts):
            class_type = class_names[idx]
            color = colors[idx]
            file_class_coords = combined_pred_dict[fname]
            for box_coords in file_class_coords:
                x1, y1, x2, y2 = box_coords
                if y1 > img_rows or y2 > img_rows:
                    print('y sizing issue for {}; coords={}'.format(fname, box_coords))
                if x1 > img_cols or x2 > img_cols:
                    print('y sizing issue for {}; coords={}'.format(fname, box_coords))
                ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1))

        plt.title('{} predictions for {}'.format(class_names, fname))
    return


def patch_stats(img_patch, visualise=False):
    patch_mean = np.mean(img_patch)
    patch_std = np.std(img_patch)
    # np.histogram
    # plt.hist(np.flatten(img_patch))
    if visualise:
        plt.figure()
        plt.imshow(img_patch)
    return patch_mean, patch_std, img_patch.shape


def plot_img_boxes(key, img, box_data, true_coords=[], color='white'):
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    for box_coords in box_data:
        x1, y1, x2, y2 = box_coords
        if y1 > img_rows or y2 > img_rows:
            print('y sizing issue for {}; coords={}'.format(key, box_coords))
        if x1 > img_cols or x2 > img_cols:
            print('y sizing issue for {}; coords={}'.format(key, box_coords))
        ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1))

    if len(true_coords)>0:
        ax1.scatter(x=true_coords[:,0], y=true_coords[:, 1], c='red', s=2)
    return


def get_true_coords(folder, visualise=False):
    json_files = os.listdir(folder)
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


def plot_img_coords(folder, fname, file_coords):
    fpath = os.path.join(folder, fname+'.png')
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    plt.figure(1)
    plt.clf()
    plt.imshow(img)
    file_coords_np = np.asarray(file_coords)
    plt.scatter(x=file_coords_np[:, 0], y=file_coords_np[:, 1], c='red', s=2)
    return img


# copied functions over from analyseBoxResults
def get_predicted_coords(folder, file_name="coords_ac.txt"):
    predicted_dict = {}
    with open("{}/{}".format(folder, file_name)) as fin:
        for l in fin:
            arr = l.rstrip().split("\t")
            file_name = arr[0].replace('.png', '')
            if file_name in predicted_dict:
                predicted_dict[file_name] += literal_eval(arr[1])
            else:
                predicted_dict[file_name] = literal_eval(arr[1])
    return predicted_dict


def get_coords(coord_file):
    fin = open(coord_file).read()
    json_data = json.loads(fin)
    return json_data


# utils
def get_file_names(folder):
    files = os.listdir(folder)
    img_files = [x for x in files if 'mask' not in x and '.png' in x]
    return img_files


def review_preds(is_traced=True):
    from analyser import predict_image_mask, combine_predicted_strips_into_image, load_params
    ac_imgs, img_names = get_traced_images(is_traced)

    pred_npy = os.path.join(base_folder, 'traced_ac_mask_preds.npy' if is_traced else 'empty_ac_mask_preds.npy')
    ac_preds = np.load(pred_npy)

    # compare against individual images
    for idx, img_name in enumerate(img_names):
        img_npy = '{}/{}_preds.npy'.format(base_folder, img_name)
        if os.path.isfile(img_npy):
            img_preds = np.load(img_npy)
        else:
            continue
        plt.figure(1)
        plt.clf()
        plt.imshow(ac_imgs[idx,])
        plt.figure(2)
        plt.clf()
        plt.imshow(ac_preds[idx, ])

        raw_rows = img_rows
        raw_cols = img_cols
        cropped_rows = 496
        cropped_cols = 128
        pixel_overlap = 8
        raw_aug = (raw_rows - cropped_rows) / pixel_overlap * (raw_cols - cropped_cols) / pixel_overlap
        img_combined = combine_predicted_strips_into_image(img_preds, 0, num_aug=int(raw_aug), img_rows=raw_rows,
                                                           img_cols=raw_cols, debug_mode=False)
        plt.figure(3)
        plt.clf()
        plt.imshow(img_combined)
    return


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(np.ceil(result.size/2)):]


def simple_corr_analysis(data):
    nrows, ncols = data.shape
    row_corrs = []
    for idx in range(nrows):
        x = data[idx, ]
        row_corrs.append(autocorr(x))

    col_corrs = []
    for idx in range(ncols):
        x = data[:, idx]
        col_corrs.append(autocorr(x))

    return row_corrs, col_corrs


def compare_img_conversions():
    # visualise some test images
    # tiffs vs png vs downsampled png
    tiff_path = './acseg/Inflamed/20170703mouse6_Day2_Right/20170703mouse6_Day2_Right (656).tiff'
    png_path = './Inflamed_201703mouse6_Day2_Right_656_from_tiff_orig.png'
    png_ds_path = './Inflamed_201703mouse6_Day2_Right_656_from_tiff.png'
    png_seg_path = './Kathryn-Inflamed_20170703mouse6_Day2_Right_656.png'
    png_seg_ds_path = './Inflamed_20170703mouse6_Day2_Right_656_ds.png'     # this is from png

    tiff_img = cv2.imread(tiff_path, cv2.IMREAD_GRAYSCALE)
    png_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    png_ds_img = cv2.imread(png_ds_path, cv2.IMREAD_GRAYSCALE)

    png_seg_img = cv2.imread(png_seg_path, cv2.IMREAD_GRAYSCALE)
    png_seg_ds_img = cv2.imread(png_seg_ds_path, cv2.IMREAD_GRAYSCALE)

    # some stats
    patch_stats(tiff_img, visualise=True)
    patch_stats(png_img, visualise=True)
    patch_stats(png_seg_img, visualise=True)
    np.sum(png_img==png_seg_img)    # these are the same

    patch_stats(png_ds_img, visualise=True)
    patch_stats(png_seg_ds_img, visualise=True)
    np.sum(png_ds_img==png_seg_ds_img)  # but scaling 50% from tiff and png are not - though similar
    np.mean(abs(png_ds_img.astype(np.float32) - png_seg_ds_img.astype(np.float32)))

    # look at higher moments
    from scipy.stats import moment
    plt.figure(1);
    plt.clf();
    plt.grid()
    # plt.title()
    # plt.hist(png_img.flatten(), bins=[1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150])
    # plt.hist([png_img.flatten(), png_ds_img.flatten(), png_seg_ds_img.flatten(), png_img[::2, ::2].flatten()],
    #          bins=[1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150])
    # plt.legend(['png from tiff', 'scaled png from tiff', 'scaled png from png', 'ds png from png'])
    plt.hist([png_ds_img.flatten(), png_seg_ds_img.flatten(), png_img[::2, ::2].flatten()],
             bins=[1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150])
    plt.legend(['scaled png from tiff', 'scaled png from png', 'ds png from png'])

    print(moment(png_img.flatten(), moment=[1, 2, 3, 4, 5]))
    print(moment(png_ds_img.flatten(), moment=[1, 2, 3, 4, 5]))
    print(moment(png_seg_ds_img.flatten(), moment=[1, 2, 3, 4, 5]))
    print(moment(png_img[::2, ::2].flatten(), moment=[1, 2, 3, 4, 5]))

    # auto correlate for some middle columns
    row_corrs, col_corrs = simple_corr_analysis(png_img)
    row_corrs, col_corrs = simple_corr_analysis(png_ds_img)
    row_corrs, col_corrs = simple_corr_analysis(png_seg_ds_img)
    row_corrs, col_corrs = simple_corr_analysis(png_img[::2, ::2])
    return


def pred_experiment():
    # tiff_path = './acseg/Inflamed/20170703mouse6_Day2_Right/20170703mouse6_Day2_Right (656).tiff'
    png_path = './Inflamed_201703mouse6_Day2_Right_656_from_tiff_orig.png'
    png_ds_path = './Inflamed_201703mouse6_Day2_Right_656_from_tiff.png'
    png_seg_path = './Kathryn-Inflamed_20170703mouse6_Day2_Right_656.png'
    png_seg_ds_path = './Inflamed_20170703mouse6_Day2_Right_656_ds.png'     # this is from png

    png_img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)    # from tiff
    png_ds_img = cv2.imread(png_ds_path, cv2.IMREAD_GRAYSCALE)  # from tiff
    png_seg_img = cv2.imread(png_seg_path, cv2.IMREAD_GRAYSCALE)    # manually segmented png
    png_seg_ds_img = cv2.imread(png_seg_ds_path, cv2.IMREAD_GRAYSCALE)  # from png
    png_naive_ds_img = png_img[::2, ::2]    # naive downsampling

    from train import get_unet
    from analyser import predict_image_mask
    model = get_unet()
    weight_path = '{}/{}'.format(results_folder, test_weights)
    model.load_weights(weight_path)

    png_ds_mask, strip_ds_preds = predict_image_mask(model, png_ds_img)
    png_seg_ds_mask, strip_seg_ds_preds = predict_image_mask(model, png_seg_ds_img)
    png_naive_ds_mask, strip_naive_dspreds = predict_image_mask(model, png_naive_ds_img)

    # grab prediction for this image in pred stack
    is_traced = True
    ac_imgs, img_names = get_traced_images(is_traced)
    pred_npy = os.path.join(base_folder, 'traced_ac_mask_preds.npy' if is_traced else 'empty_ac_mask_preds.npy')
    ac_preds = np.load(pred_npy)

    pred_idx = img_names.index(png_seg_path.replace('./', ''))
    plt.figure()
    plt.imshow(ac_preds[pred_idx, ])
    return


def count_num_volume(folder):
    files = os.listdir(folder)
    vol_name_dict = {}
    for file in files:
        if '.png' in file and 'mask' not in file:
            vol_name = '_'.join(file.split('_')[1:-1])
            if vol_name not in vol_name_dict:
                vol_name_dict[vol_name] = [file]
            else:
                vol_name_dict[vol_name].append(file)
    return vol_name_dict, len(vol_name_dict.keys())


def combined_patch_class_preds(is_traced=True):
    # get patch data for plotting purposes
    all_patch_data, img_names = load_patch_npy(is_traced)

    # read coords file into dictionary
    true_dict = get_true_coords(json_folder)
    # parse coords file
    pred_dict_cell = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'), 'coords_cell.txt')
    pred_dict_lite = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'), 'coords_cell_lite.txt')
    pred_dict_med = get_predicted_coords(os.path.join(base_folder, 'seg_accell_img_patches'), 'coords_cell_medium.txt')
    # visualize_all_preds(all_patch_data, img_names, pred_dict, true_dict)


    # combined_pred_dict = recombine_predictions(pred_dict, all_patch_data, img_names)    # combined predicted coords
    combined_pred_dict_cell = recombine_predictions(pred_dict_cell, all_patch_data, img_names=img_names)  # combined predicted coords
    visualise_pred_vs_truth(true_dict, combined_pred_dict_cell, class_type='cell')
    combined_pred_dict_lite = recombine_predictions(pred_dict_lite, all_patch_data, img_names=img_names)  # combined predicted coords
    visualise_pred_vs_truth(true_dict, combined_pred_dict_lite, class_type='cell_lite')
    combined_pred_dict_med = recombine_predictions(pred_dict_med, all_patch_data, img_names=img_names)  # combined predicted coords
    visualise_pred_vs_truth(true_dict, combined_pred_dict_med, class_type='cell_medium')

    # plot all classes together
    visualise_preds_multi_class(true_dict, [combined_pred_dict_cell, combined_pred_dict_med], class_names=['cell', 'cell_medium'])
    visualise_preds_multi_class(true_dict, [combined_pred_dict_cell, combined_pred_dict_med, combined_pred_dict_lite], class_names=['cell', 'cell_medium', 'cell_lite'])
    # pred closeness based on predicted and actual
    # precision/sensitivity measures overall
    # precision/sensitivity measures within seg chamber
    return


if __name__ == '__main__':
    ## for arvo abstract
    # count_num_volume(folder=orig_img_folder)
    # count_num_volume(folder=empty_img_folder)

    # create_pred_img_patches(is_traced=True)
    # get_ac_preds_for_images(is_traced=True)
    # review/visualise ac cell preds for images
    # combined_patch_preds()
    combined_patch_class_preds()

    ### review segmentation and understand differences
    # accell_imgs = get_file_names('./accell/segmentations')
    # acseg_images = get_file_names('./acseg/segmentations')
    # acseg_empty_images = get_file_names('./acseg/empty_segmentations')

    # compare resolutions and img intensity moment profiles
    # compare_img_conversions()
    # thought experiment on predicting whole image vs parts of image
    # pred_experiment()
    # review_preds()
    1